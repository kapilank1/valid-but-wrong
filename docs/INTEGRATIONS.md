# Integrations: what you provide, where it goes, what to run

Three things need a human: a Vultr API key, a Tiger Data connection string, and a Cartesia API key. Everything else is
scripted. All three go in gitignored files (`.gitignore` matches `.env` and `.env.local` at any depth, and
`apps/console/.gitignore` also covers `.env*.local`). They are never printed, committed, or put in Vultr user-data.

| What | Get it from | Put it in | Used by |
|---|---|---|---|
| `DATABASE_URL` | Tiger Cloud: create a service (free trial), copy its connection string `postgres://tsdbadmin:<password>@<id>.<project>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require` | repo-root `.env` | `scripts/serve.py`, `scripts/db_init.py`, `scripts/deploy_vultr.ps1` (copies it to the server) |
| `CARTESIA_API_KEY` | play.cartesia.ai → API keys | `apps/console/.env.local` | the console's `/api/tts` route (voice callouts); `deploy_vultr.ps1` copies it to the server |
| `VULTR_API_KEY` | my.vultr.com → Account → API (enable it; allow your IP or "any") | repo-root `.env` | `scripts/deploy_vultr.ps1`, `scripts/deploy_vultr_destroy.ps1` |

Repo-root `.env` (create it; it is ignored by git):

```
DATABASE_URL=postgres://tsdbadmin:...@....tsdb.cloud.timescale.com:3xxxx/tsdb?sslmode=require
VULTR_API_KEY=...
```

`apps/console/.env.local` (copy from `apps/console/.env.example`):

```
CARTESIA_API_KEY=...
NEXT_PUBLIC_VBW_API=http://localhost:8000
```

**Cost:** the demo instance is `vc2-2c-4gb` in `sjc`, $0.027/hour (about $0.65/day, $20/month cap), billed until you
run the destroy script (price read from Vultr's public plan list on 2026-10-24; `-DryRun` shows the current one).
Tiger Cloud has a free trial and Cartesia a free tier; check their current limits.

## 1. Tiger Data (about 2 minutes)

```powershell
uv run python scripts/db_init.py --check
```

This applies the schema (two hypertables, the `verdicts_1s` continuous aggregate, refresh and compression policies),
then writes a synthetic run through the same store code the server uses. It reads the run back through the dashboard
queries and deletes it. Every line should say `[ok]`. Then restart the API (`uv run python scripts/serve.py`).
`GET http://localhost:8000/api/health` shows `"db": {"available": true, ...}`, and every run is recorded:
`GET /api/runs`, `/api/runs/<id>/timeline`, `/api/analytics`.

On Windows, the server talks to the database through sync psycopg on a worker thread, because psycopg's async driver
can't run on the ProactorEventLoop that uvicorn uses there. This happens automatically.

## 2. Cartesia (about 1 minute)

Put the key in `apps/console/.env.local` and restart `pnpm --dir apps/console dev`.
`GET http://localhost:3000/api/tts` returns `{"configured": true, ...}`. Alerts are then spoken in the Cartesia voice
(`sonic-3.6`, `Cartesia-Version: 2026-08-14`, mp3); without a key the console falls back to the browser's voice.
Optional overrides: `CARTESIA_VOICE_ID`, `CARTESIA_MODEL_ID`.

## 3. Vultr (about 15 minutes, mostly the first Docker build)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deploy_vultr.ps1 -DryRun   # shows plan, price, API calls; creates nothing
powershell -ExecutionPolicy Bypass -File scripts\deploy_vultr.ps1 -Yes      # creates the instance and deploys
```

At the end it prints the console URL (`http://<ip>:3000`) and the API URL (`http://<ip>:8000/api/health`).
With `DATABASE_URL` set, the server writes to Tiger Cloud. Without it, the instance runs its own TimescaleDB
container. Details are in [DEPLOY_VULTR.md](DEPLOY_VULTR.md).

- New code or models: `scripts\deploy_vultr.ps1 -Update`
- HTTPS on a domain: point its A record at the IP, then `scripts\deploy_vultr.ps1 -Update -Domain yourname.tech`
- Stop billing: `scripts\deploy_vultr_destroy.ps1 -Yes`

## Verified with real credentials (2026-10-24)

- **Tiger Data:** `db_init.py --check` passed against a Tiger Cloud service (TimescaleDB 2.31.0): schema, both
  hypertables, `verdicts_1s` + refresh/compression jobs, synthetic round trip. A live ▶ Demo run then wrote 8,961 rows
  in 64 batches with 0 errors; `/api/runs`, `/api/runs/<id>/timeline` (300 one-second buckets) and `/api/analytics`
  all answered from Tiger Cloud.
- **Cartesia:** the console's `/api/tts` returned MP3 (140 KB, ~2 s first call, then cached); a live ▶ Demo spoke all
  4 alerts through Cartesia (4 × `POST /api/tts` 200).
- **TLS-inspecting antivirus (Norton on the dev laptop):** Node's bundled CA list doesn't include the antivirus root,
  so server-side `fetch` to Cartesia failed with `UNABLE_TO_VERIFY_LEAF_SIGNATURE`. The console now also trusts the OS
  certificate store at server start (`apps/console/src/instrumentation.ts`, `src/lib/systemCa.ts`). git uses
  schannel (OS store) and psycopg uses `sslmode=require`, so neither was affected.
- **Vultr:** skipped for now (no credits).

## Not verified yet

These scripts were written and checked without credentials:
- Syntax: the PowerShell parser and `bash -n`.
- Unit tests for the store, including the threaded driver.
- A Vultr `-DryRun` against the live public catalog (plan, price, OS id).
- A local build of the upload bundle.

Still untested (Vultr only):
- Real instance creation.
- The cloud-init run.
- Building the Docker images on the host.

The first real run of each will show any problem.
