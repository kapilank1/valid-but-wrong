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
