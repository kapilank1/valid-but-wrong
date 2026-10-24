# Deploying on Vultr

There are two machines. A **Cloud Compute** instance hosts the live demo (TimescaleDB or Tiger Cloud, the pipeline
server and the console). A **Cloud GPU** instance trains the models; you copy the results back and then destroy it.

## 0. One-time setup (laptop)

```bash
brew install vultr/vultr-cli/vultr-cli
export VULTR_API_KEY=...                      # Vultr account -> API
vultr-cli ssh-key create --name vbw --key "$(cat ~/.ssh/id_ed25519.pub)"
vultr-cli ssh-key list                        # note the SSH key id
vultr-cli regions list                        # e.g. sjc (Silicon Valley), lax
vultr-cli os list | grep -i "ubuntu 24.04"    # note the OS id
```

## 1. Demo host (Cloud Compute)

```bash
vultr-cli plans list --type vc2 | grep -E "4c-8gb|4c-16gb"    # 4 vCPU / 8 GB is enough
vultr-cli instance create --region sjc --plan vc2-4c-8gb --os <OS_ID> \
  --ssh-keys <KEY_ID> --label vbw-demo --host vbw-demo
vultr-cli instance list                      # wait for "active", note MAIN IP
ssh root@<IP>
```

On the host:

```bash
curl -fsSL https://get.docker.com | sh
ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw allow 3000/tcp && ufw allow 8000/tcp && ufw --force enable
git clone <repo-url> valid-but-wrong && cd valid-but-wrong
python3 scripts/fetch_smap.py                # SMAP/MSL scenarios (data/raw is gitignored); stdlib only
cp infra/.env.example infra/.env
```

Edit `infra/.env`:

```bash
NEXT_PUBLIC_VBW_API=http://<IP>:8000         # what the browser calls
VBW_CORS_ORIGINS=http://<IP>:3000
CARTESIA_API_KEY=...
DATABASE_URL=                                # empty = use the bundled TimescaleDB; or paste Tiger Cloud (below)
```

Start everything and check it:

```bash
docker compose -f infra/docker-compose.yml up -d --build          # first build is slow: torch wheels are big
curl http://<IP>:8000/api/health                                 # {"ok":true,"db":{"available":true,...}}
open http://<IP>:3000
```

**HTTPS + one origin (optional).** Point a DNS A record (e.g. `validbutwrong.tech`) at the IP, then set
`VBW_DOMAIN=validbutwrong.tech` and `NEXT_PUBLIC_VBW_API=https://validbutwrong.tech` in `infra/.env` and run:

```bash
docker compose -f infra/docker-compose.yml --profile edge up -d --build
```

Caddy gets the certificate, serves the console on `/` and the pipeline on `/api/*` and `/ws` (the console's own
`/api/tts` still goes to the console). No CORS setup is needed.

**Updating models without a rebuild.** `models/`, `results/` and `data/` are mounted read-only into the server.
Copy in the new files and run `docker compose -f infra/docker-compose.yml restart server`.

## 2. Tiger Data (instead of the bundled TimescaleDB)

1. Create a service at console.cloud.timescale.com and copy its connection string
   (`postgres://tsdbadmin:...@<id>.<project>.tsdb.cloud.timescale.com:<port>/tsdb?sslmode=require`).
2. Apply the schema from your laptop:
   `DATABASE_URL='<conn>' uv run python scripts/db_init.py --check`. This lists the hypertables, the
   `verdicts_1s` continuous aggregate, and the refresh and compression jobs.
3. Put the same `DATABASE_URL` in `infra/.env` and run only the app containers:
   `docker compose -f infra/docker-compose.yml up -d --build --no-deps server console`.
4. Showcase queries (served by the pipeline):
   - `GET /api/runs`: recent runs with peak score, alert, attacked and false-alert frames (from `verdicts_1s`)
   - `GET /api/runs/<run_id>/timeline`: peak score per second, plus attack onset, first alert and detection latency
   - `GET /api/analytics`: time to first alert per (attack, detector) across all runs, and the detector scorecard
     (false-alert rate on clean frames, alert rate on attacked frames)

## 3. Training on a Cloud GPU

```bash
vultr-cli plans list --type vcg              # GPU plans (A16 / A40 / L40S / A100 slices); pick one with many vCPUs
vultr-cli instance create --region <region-with-that-plan> --plan <GPU_PLAN_ID> --os <OS_ID> \
  --ssh-keys <KEY_ID> --label vbw-train
ssh root@<GPU_IP>
nvidia-smi || (apt-get update && apt-get install -y ubuntu-drivers-common && ubuntu-drivers install && reboot)
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone <repo-url> valid-but-wrong && cd valid-but-wrong
uv sync                                      # the lock's linux torch wheel bundles CUDA
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python3 scripts/fetch_smap.py
```

Run long jobs under `tmux` so a dropped SSH session doesn't kill them:

```bash
tmux new -s train
uv run python scripts/calibrate_l1.py --schema falcon9
uv run python scripts/calibrate_l1.py --schema quad
uv run python scripts/train_l2.py --schema falcon9
uv run python scripts/train_l2.py --schema quad
uv run python scripts/train_ppo.py --schema falcon9 --detector base --timesteps 1000000 --n-envs 16
uv run python scripts/make_forged_dataset.py --schema falcon9
uv run python scripts/run_bench.py --workers 16
```

Notes:
- `train_ppo.py` pins SB3 to `device="cpu"` (a small MLP policy is limited by environment stepping, not by the
  GPU). Its speed comes from vCPUs, so scale `--n-envs` to the instance's core count.
- Check each script's `--help` for the device it trains on before paying for a large GPU.

Copy the artifacts back, then **destroy the GPU instance** (it bills by the hour):

```bash
# on the laptop
rsync -av root@<GPU_IP>:valid-but-wrong/models/ models/
rsync -av root@<GPU_IP>:valid-but-wrong/results/ results/
rsync -av root@<GPU_IP>:valid-but-wrong/data/cache/ data/cache/      # forged datasets, for L3 training
vultr-cli instance delete <GPU_INSTANCE_ID>
# push the models to the demo host
rsync -av models results root@<IP>:valid-but-wrong/ && ssh root@<IP> \
  'cd valid-but-wrong && docker compose -f infra/docker-compose.yml restart server'
```
