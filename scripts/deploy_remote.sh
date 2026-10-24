#!/usr/bin/env bash
# Runs ON the Vultr host, uploaded and started over SSH by scripts/deploy_vultr.ps1. Puts the code in /opt/vbw,
# installs infra/.env, makes sure the SMAP/MSL demo channels exist, builds and starts the stack, waits for health.
#
#   bash deploy_remote.sh --bundle /root/vbw.tgz   [--env /root/vbw.env] [--local-db] [--edge]
#   bash deploy_remote.sh --git <url> [--branch main] [--env /root/vbw.env] [--local-db] [--edge]
#
#   --local-db   also run the TimescaleDB container (no Tiger Cloud DATABASE_URL given)
#   --edge       also run Caddy (compose profile "edge") for HTTPS on $VBW_DOMAIN
set -euo pipefail

APP=/opt/vbw
BUNDLE="" GIT_URL="" BRANCH=main ENV_FILE=/root/vbw.env LOCAL_DB=0 EDGE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2 ;;
    --git) GIT_URL="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --env) ENV_FILE="$2"; shift 2 ;;
    --local-db) LOCAL_DB=1; shift ;;
    --edge) EDGE=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[vbw %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "waiting for cloud-init (Docker install) to finish"
cloud-init status --wait >/dev/null 2>&1 || true
for _ in $(seq 1 90); do docker info >/dev/null 2>&1 && break; sleep 2; done
docker info >/dev/null 2>&1 || { echo "docker is not running on this host" >&2; exit 1; }

mkdir -p "$APP"
if [ -n "$BUNDLE" ]; then
  log "unpacking $BUNDLE into $APP"
  tar --warning=no-unknown-keyword -xzf "$BUNDLE" -C "$APP"   # bundle made by Windows bsdtar: skip its pax keywords
elif [ -n "$GIT_URL" ]; then
  if [ -d "$APP/.git" ]; then
    log "updating $APP from $GIT_URL ($BRANCH)"
    git -C "$APP" fetch --depth 1 origin "$BRANCH"
    git -C "$APP" reset --hard FETCH_HEAD
  else
    log "cloning $GIT_URL ($BRANCH) into $APP"
    git clone --depth 1 --branch "$BRANCH" "$GIT_URL" "$APP"
  fi
else
  echo "need --bundle or --git" >&2; exit 2
fi
cd "$APP"

if [ -f "$ENV_FILE" ]; then
  install -m 600 "$ENV_FILE" infra/.env
  rm -f "$ENV_FILE"
fi
[ -f infra/.env ] || { echo "infra/.env is missing (deploy_vultr.ps1 uploads it)" >&2; exit 1; }

if [ ! -f data/raw/smap_msl/labeled_anomalies.csv ]; then
  log "SMAP/MSL data missing: fetching it (scripts/fetch_smap.py, stdlib only)"
  python3 scripts/fetch_smap.py || log "WARNING: SMAP/MSL fetch failed; spacecraft scenarios will be unavailable"
fi

COMPOSE=(docker compose -f infra/docker-compose.yml)
SERVICES=(server console)
[ "$EDGE" = 1 ] && COMPOSE+=(--profile edge) && SERVICES+=(caddy)
if [ "$LOCAL_DB" = 1 ]; then
  log "building and starting db + ${SERVICES[*]} (first build takes several minutes)"
  "${COMPOSE[@]}" up -d --build db "${SERVICES[@]}"
else
  log "building and starting ${SERVICES[*]} against the external DATABASE_URL (first build takes several minutes)"
  "${COMPOSE[@]}" up -d --build --no-deps "${SERVICES[@]}"
fi

log "waiting for the API and the console to answer"
api_ok=0 console_ok=0
for _ in $(seq 1 120); do
  [ "$api_ok" = 1 ] || { curl -fsS -m 3 http://127.0.0.1:8000/api/health >/dev/null 2>&1 && api_ok=1; }
  [ "$console_ok" = 1 ] || { curl -fsS -m 3 -o /dev/null http://127.0.0.1:3000/ 2>/dev/null && console_ok=1; }
  [ "$api_ok" = 1 ] && [ "$console_ok" = 1 ] && break
  sleep 5
done
"${COMPOSE[@]}" ps
if [ "$api_ok" = 1 ]; then
  log "API health: $(curl -fsS -m 3 http://127.0.0.1:8000/api/health)"
else
  log "API did not come up; last server logs:"; "${COMPOSE[@]}" logs --tail 60 server || true
fi
[ "$console_ok" = 1 ] || { log "console did not come up; last console logs:"; "${COMPOSE[@]}" logs --tail 60 console || true; }
[ "$api_ok" = 1 ] && [ "$console_ok" = 1 ]
