# Valid-But-Wrong · mission console

Next.js (App Router) mission-control UI for Valid-But-Wrong. It shows truth vs reported telemetry, link-layer (L0)
integrity, detector verdicts (L1 physics, L2 learned, L3 hardened), alerts with voice callouts, the attack ladder,
and SMAP/MSL bench numbers. It speaks the WS/REST contract in [`docs/CONTRACT.md`](../../docs/CONTRACT.md), and
`src/lib/types.ts` mirrors `vbw/schema.py`.

## Run

```bash
cd apps/console
pnpm install
cp .env.example .env.local      # optional: set CARTESIA_API_KEY / NEXT_PUBLIC_VBW_API
pnpm dev                        # http://localhost:3000
```

Other scripts: `pnpm build` / `pnpm start` (production), `pnpm test` (vitest), `pnpm typecheck`.

## Env vars

| Var | Where | Default | Purpose |
|---|---|---|---|
| `NEXT_PUBLIC_VBW_API` | client | `http://localhost:8000` | Pipeline server base URL. The WebSocket is `<base>/ws`. |
| `CARTESIA_API_KEY` | server only | (unset) | Enables Cartesia voice callouts via `POST /api/tts`. Unset → browser `speechSynthesis`. |
| `CARTESIA_MODEL_ID` | server only | `sonic-3.6` | Cartesia model. |
| `CARTESIA_VOICE_ID` | server only | Cartesia example voice | Voice id. |
| `CARTESIA_VERSION` | server only | `2026-08-14` | `Cartesia-Version` header. |

`NEXT_PUBLIC_*` values are inlined at build time, so rebuild after changing them.

## Mock mode

Open `http://localhost:3000/?mock=1` to force it. The console also falls back to it automatically if
`GET /api/health` doesn't answer within 1.5 s. A **MOCK** badge in the top bar says which case you're in, and
"Retry live" reconnects once the server is up.

The mock (`src/lib/mock/`) is an in-browser imitation of the pipeline server with the same contract:
- **Falcon 9:** a real SES-9 webcast ascent, downsampled to 1 Hz (`pnpm gen:mock-profile [slug]` regenerates it from
  `data/processed/falcon9`). There are also a synthetic quad figure-8 and a synthetic SMAP-like channel with labeled
  anomaly ranges.
- **Attacks:** bias, drift, replay, coordinated, ppo. Every forged frame stays L0-valid.
- **Detector:** L1 runs real consistency rules on the reported stream. L2/L3 are faked from the truth gap and tuned
  to the demo script: bias is caught instantly by L1, drift by L2 after ~12 s, PPO evades `base` and is caught after
  you hot-swap to `hardened`.
- **Bench:** the bench card is labeled MOCK DATA. Its baseline row is Telemanom's published numbers.

## Live behavior

- WebSocket reconnects with exponential backoff (0.5 s → 8 s, jittered). The pill shows LIVE / RECONNECTING.
- Ticks go into ring buffers outside React. Canvases redraw on one shared `requestAnimationFrame` loop, and only
  when data or size changed. Text panels get throttled snapshots at ~5 Hz. Alerts and status messages show
  immediately.
- Attack ladder buttons call `POST /api/attack` with `start_t` = now. A rung is disabled if the catalog lacks the
  attack, marks it `available: false`, or the attack's `schemas` excludes the current vehicle. "Harden detector" and
  the top-bar toggle call `POST /api/detector`.
- Scenario, attack config, detector and speed are sent with **Start** / **Reset** (`POST /api/control`). The contract
  has no speed-only command, so speed applies on the next Start/Reset.
- Voice: on each WS `alert`, the `spoken` line is queued. Callouts never overlap and stale ones are dropped. Cartesia
  goes through `/api/tts` (mp3, cached in memory per line), with `speechSynthesis` as the fallback. There is a mute
  toggle. Browsers only allow audio after a click on the page, so press Start (or anything) first.

Layout targets a 1440×900 projector with no page scroll. Below 1100 px wide it stacks and scrolls.
