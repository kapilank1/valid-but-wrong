"""Load report for the fleet workers running in the cluster.

    uv run python scripts/fleet_report.py                  # one 15 s window
    uv run python scripts/fleet_report.py --watch          # repeat until Ctrl-C
    uv run python scripts/fleet_report.py --out results/fleet_scale.json --label replicas-6

Scrapes every worker pod's /metrics through the API server's pod proxy (`kubectl get --raw`), so it needs no
port-forward and no Prometheus. Throughput is measured from counter deltas over the window.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vbw.fleet.report import FleetSummary, Samples, parse, summarize  # noqa: E402


def kubectl(*args: str) -> str:
    return subprocess.run(["kubectl", *args], check=True, capture_output=True, text=True).stdout


def worker_pods(namespace: str) -> list[str]:
    pods = json.loads(kubectl("-n", namespace, "get", "pods", "-l", "app.kubernetes.io/name=worker", "-o", "json"))
    return [p["metadata"]["name"] for p in pods["items"]
            if any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))]


def scrape(namespace: str, pods: list[str]) -> dict[str, Samples]:
    out = {}
    for pod in pods:
        try:
            out[pod] = parse(kubectl("get", "--raw", f"/api/v1/namespaces/{namespace}/pods/{pod}:9100/proxy/metrics"))
        except subprocess.CalledProcessError:
            pass   # pod went away mid-scrape (scale down); it drops out of this window
    return out


def show(s: FleetSummary, seconds: float) -> None:
    print(f"\n{'pod':<10} {'vehicles':>8} {'frames/s':>10} {'needed':>8} {'lag s':>8} {'lag s/s':>8}")
    for p in s.pods:
        print(f"{p.pod:<10} {p.vehicles:>8} {p.fps:>10.0f} {p.target_fps:>8.0f} {p.lag_s:>8.1f} {p.lag_trend:>+8.2f}")
    state = "keeping up" if s.keeping_up else "FALLING BEHIND"
    print(f"{'total':<10} {s.vehicles:>8} {s.fps:>10.0f} {s.target_fps:>8.0f}   {state}  ({seconds:.0f} s window)")
    if s.p50_ms is not None:
        print(f"per-frame scoring latency: p50 {s.p50_ms:.2f} ms, p99 {s.p99_ms:.2f} ms")
    if s.flights_attacked or s.flights_clean:
        print(f"finished flights: {s.flights_detected}/{s.flights_attacked} attacked detected, "
              f"{s.false_alerts} with a false alert ({s.flights_clean} clean)"
              + (f", median detection {s.detection_p50_s:.1f} s after onset" if s.detection_p50_s else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", default="argus")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--out", type=Path, help="append this window's summary to a JSON list")
    ap.add_argument("--label", default="", help="tag stored with --out (e.g. replicas-6)")
    args = ap.parse_args()

    while True:
        pods = worker_pods(args.namespace)
        if not pods:
            sys.exit(f"no ready worker pods in namespace {args.namespace!r}")
        before, start = scrape(args.namespace, pods), time.monotonic()
        time.sleep(args.seconds)
        after = scrape(args.namespace, pods)
        elapsed = time.monotonic() - start
        summary = summarize(before, after, elapsed)
        show(summary, elapsed)
        if args.out:
            runs = json.loads(args.out.read_text()) if args.out.exists() else []
            runs.append({"label": args.label, "window_s": round(elapsed, 1), "keeping_up": summary.keeping_up,
                         "vehicles": summary.vehicles, "frames_per_s": round(summary.fps),
                         "needed_frames_per_s": round(summary.target_fps), **asdict(summary)})
            args.out.write_text(json.dumps(runs, indent=2) + "\n")
        if not args.watch:
            break


if __name__ == "__main__":
    main()
