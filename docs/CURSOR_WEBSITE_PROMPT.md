# Prompt for Cursor (Agent mode, Grok or any model): build the project website

Paste everything in the block below into Cursor's Agent chat with this repo open.

```text
You are building the public project website for "Valid-But-Wrong", our Cal Hacks 13.0 project, so judges can understand
the project and our code in a few minutes. The site has to look outstanding AND be 100% accurate.

STEP 0 — Load context before writing anything. Read, in this order:
1. .cursor/rules/project.mdc
2. CONTINUE_HERE.md (top "NEXT TASK" + "Latest" blocks)
3. docs/WEBSITE_BRIEF.md  ← the single source of truth for the site: story, every number with its source file,
   suggested sections, visual direction, data plan, and rules. Follow it.
4. README.md, docs/DEVPOST.md, docs/BENCH.md, docs/TENSORRT.md, docs/INTEGRATIONS.md, edge/SPEC.md (status block)
5. The result files the numbers come from: results/detector_eval_r2.json, results/arms_race.json,
   results/trt_bench.json, results/bench_summary.json, results/edge_bench.json
6. Skim the code so the site explains it correctly: vbw/link/codec.py, vbw/detector/{physics,l2,l3,alerting,base}.py,
   vbw/attacker/{scripted,env}.py, apps/console/src (the mission console the site should visually echo).

STEP 1 — Plan, then split the work. Write a short plan (sections, components, data files, tech) and show it to me.
If you can run parallel agents / background agents / subagents, use them with YOU as the orchestrator, one owner per
file area, for example:
  - Agent "data": scripts/export_site_data.py (per-frame SES-9 replay: true vs reported altitude, base vs hardened
    score + alerts, PPO attack from T+190 s; optionally bias at T+60 s) → apps/site/public/data/*.json, plus copying
    the results/*.json the site uses into apps/site/public/data/ (a script, not by hand). Follow the "Data for the
    site" section of the brief exactly (it names the functions to call). Run it with `uv run python ...`.
  - Agent "design system + layout": apps/site scaffold, theme (dark mission control; cyan = truth, amber = reported,
    red = alert, green = link valid; monospaced telemetry numerals), nav, hero, footer, responsive layout.
  - Agent "sections": the content sections from the brief (the attack, four detection layers, attack ladder, arms
    race, results tables, engineering depth: TensorRT / C++ edge monitor / NASA benchmark / tests, architecture
    diagram, sponsor tech, honest limits, team + links).
  - Agent "interactive": the showpiece interactive SES-9 replay (scrub through the flight; truth vs reported altitude
    with the gap shaded as "the lie"; base vs hardened detector scores with threshold line and alert markers), the
    animated "every check passes ✓ … altitude wrong by 3.7 km" packet, and the arms-race chart.
  - Agent "QA": audits every number on the site against the source files (list each number → file → value), checks
    claims against the brief's "don't claim" list, mobile layout (375 px), 1440 px layout, keyboard navigation,
    prefers-reduced-motion, contrast, and build/lint.
If you can't run parallel agents, do the same work yourself in that order.

STEP 2 — Build.
- Put the site in apps/site/ as its own app: Next.js (App Router) + TypeScript + Tailwind, static export
  (output: "export"), pnpm. Light dependencies only (hand-rolled SVG/canvas charts or one small chart library;
  framer-motion optional). No backend, no tracking.
- Numbers are loaded from the JSON files in apps/site/public/data (copied from results/), not retyped, wherever
  practical. Anything hand-written must match the brief/results exactly.
- Render the architecture diagram (README "How it works" Mermaid) as a clean, themed SVG.
- Do NOT modify vbw/, apps/console/, edge/, models/, results/ or any experiment code. Never read .env or
  apps/console/.env.local. Don't invent team members, users, quotes or metrics. Respect the brief's limits section
  and "don't claim" list (no Kubernetes, no live Vultr deployment, no live ArduPilot, TensorRT is a benchmark not the
  live path, below Telemanom on the NASA benchmark).
- Ask me for anything only I have: team member names, Devpost URL, demo video URL, and console screenshots (or capture
  screenshots yourself from the running console at http://localhost:3000 if you have a browser tool: start it with
  `uv run python scripts/serve.py` and `pnpm --dir apps/console dev`, click "▶ Demo"). Use clearly marked
  placeholders until I provide them.

STEP 3 — Verify and hand off.
- `pnpm --dir apps/site build` must pass; run it locally and check it at 375 px and 1440 px.
- QA's number audit table must show every number matching its source.
- Add a short apps/site/README.md (run, build, regenerate data, deploy to Vercel or GitHub Pages) and a "Website"
  line in the root README.md.
- Commit in small steps with clear messages. Ask me before pushing to GitHub or deploying anywhere.
- Append what you did to HANDOFF.md and update the top of CONTINUE_HERE.md.

Environment notes: Windows + PowerShell 5.1 (no &&; use ;). Python via `uv run`; uv network installs need
`--system-certs` (Norton intercepts TLS). Node 26 + pnpm. git is at "C:\Program Files\Git\cmd\git.exe" if `git`
isn't on PATH. Deadline: Sun Oct 25 2026, 9:00 AM PST — keep scope tight: a polished, accurate single-page site with
one excellent interactive replay beats many half-done pages.

Start with STEP 0 and STEP 1 (the plan), then proceed.
```
