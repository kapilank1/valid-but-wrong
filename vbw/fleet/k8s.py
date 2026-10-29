"""Find the live fleet workers through the Kubernetes API, using the pod's own service-account token.

Every worker lists the worker pods and keeps the ordinals of the live ones; that set is what vehicles are sharded
over (see sharding.py). Live = not shutting down, and either Ready or with a container that started in the last
STARTING_GRACE_S seconds. The grace covers pods that start together: none is Ready yet, and without it each one
would see only itself and take the whole fleet (that OOM-killed every worker on a cold start). A crash-looping
pod has no running container, so it drops out at once; a hung one drops out when its readiness probe fails.

The pod is allowed to list pods by the Role in infra/k8s/worker.yaml (nothing else). Plain urllib so the worker
needs no Kubernetes client library.
"""

from __future__ import annotations

import json
import ssl
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from vbw.fleet.sharding import ordinal_from_hostname

SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
API = "https://kubernetes.default.svc"
STARTING_GRACE_S = 30.0

Fetch = Callable[[urllib.request.Request, ssl.SSLContext], dict]


def in_cluster(sa_dir: Path = SA_DIR) -> bool:
    return (sa_dir / "token").exists()


def _is_live(pod: dict, now: datetime, grace_s: float) -> bool:
    if pod["metadata"].get("deletionTimestamp"):      # terminating pods can still report Ready
        return False
    status = pod.get("status", {})
    if any(c["type"] == "Ready" and c["status"] == "True" for c in status.get("conditions", [])):
        return True
    for container in status.get("containerStatuses", []):
        running = container.get("state", {}).get("running")
        if running and not container.get("ready"):
            started = datetime.fromisoformat(running["startedAt"].replace("Z", "+00:00"))
            if (now - started).total_seconds() < grace_s:
                return True
    return False


def live_ordinals(pod_list: dict, now: datetime | None = None, grace_s: float = STARTING_GRACE_S) -> set[int]:
    now = now or datetime.now(timezone.utc)
    out = set()
    for pod in pod_list.get("items", []):
        if _is_live(pod, now, grace_s):
            try:
                out.add(ordinal_from_hostname(pod["metadata"]["name"]))
            except ValueError:
                pass
    return out


def live_workers(label_selector: str, sa_dir: Path = SA_DIR, fetch: Fetch | None = None) -> set[int]:
    namespace = (sa_dir / "namespace").read_text().strip()
    token = (sa_dir / "token").read_text().strip()   # re-read every call: the kubelet rotates it
    query = urllib.parse.urlencode({"labelSelector": label_selector})
    request = urllib.request.Request(
        f"{API}/api/v1/namespaces/{namespace}/pods?{query}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    context = ssl.create_default_context(cafile=str(sa_dir / "ca.crt"))
    return live_ordinals((fetch or _fetch)(request, context))


def _fetch(request: urllib.request.Request, context: ssl.SSLContext) -> dict:
    with urllib.request.urlopen(request, context=context, timeout=5) as response:
        return json.load(response)
