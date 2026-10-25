"""End-to-end evaluation of the C++ edge monitor against the Python BaseDetector, on the same UDP frames.

    uv run python scripts/edge_eval.py                       # every case -> results/edge_eval.json
    uv run python scripts/edge_eval.py --cases bias-max,fleet8-rt50
    uv run python scripts/edge_eval.py --python-only         # no exe needed -> results/edge_eval_python_only.json
    uv run python scripts/edge_eval.py --list

Per case: precompute every vehicle's packets with scripts/udp_replay.py (attack applied, valid CRC), launch
`edge/build/vbw_edge.exe --config edge/config/falcon9.cfg --port P --jsonl <tmp> --log-frames --idle-exit 3`, send the
packets over loopback UDP (paced or flat out), wait for the monitor's idle exit, then join its JSONL with the ground
truth. The same packets also go through the Python ground side (decode + LinkMonitor + BaseDetector) in-process, so
both columns of the report judge byte-identical input:

- per vehicle: detected?, onset -> first alert latency (mission s), false alerts (rising edges before the onset, or
  any on a clean vehicle), for C++ and Python side by side;
- agreement: are the alert rising-edge times equal, how many frames' alert flag differs, the largest score gap;
- C++ per-frame processing latency p50/p99/max (recvfrom -> verdict, from the frame lines), frames/s, frames dropped
  (sent vs received); Python per-frame latency for comparison.

The binary is built separately (edge/build.ps1); this script builds nothing. Python results don't depend on the send
speed, so they are computed once per distinct stream and reused across cases.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import socket
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import udp_replay  # noqa: E402

from vbw.detector.registry import make_detector  # noqa: E402
from vbw.link.codec import LinkMonitor  # noqa: E402
from vbw.schema import SCHEMAS_BY_NAME  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EXE = ROOT / "edge" / "build" / "vbw_edge.exe"
CONFIG = ROOT / "edge" / "config" / "falcon9.cfg"
T_MATCH = 1e-3        # alert edge times are frame times 0.1 s apart; this only absorbs JSON print precision

# name -> overrides of CASE_DEFAULTS. "max" = --speed 0 (flat out), "rt50" = real time at 50 Hz per vehicle.
CASE_DEFAULTS: dict[str, Any] = dict(scenario=udp_replay.DEFAULT_SCENARIO, vehicles=1, attacked=None, seed=0,
                                     attack="none", attack_start=udp_replay.DEFAULT_ATTACK_START, intensity=1.0,
                                     speed=0.0, rate_hz=None)
CASES: dict[str, dict[str, Any]] = {
    "clean-max": dict(attack="none", speed=0.0),
    "clean-rt50": dict(attack="none", speed=1.0, rate_hz=50.0),
    "bias-max": dict(attack="bias", speed=0.0),
    "bias-rt50": dict(attack="bias", speed=1.0, rate_hz=50.0),
    "ppo-max": dict(attack="ppo", speed=0.0),
    "ppo-rt50": dict(attack="ppo", speed=1.0, rate_hz=50.0),
    "fleet8-rt50": dict(vehicles=8, attacked=4, attack="bias", speed=1.0, rate_hz=50.0),
    "fleet8-max": dict(vehicles=8, attacked=4, attack="bias", speed=0.0),
}


# ---------------------------------------------------------------------------------------------- helpers


def percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"p50": None, "p99": None, "max": None, "mean": None}
    a = np.asarray(values, dtype=float)
    return {"p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)), "max": float(a.max()),
            "mean": float(a.mean())}


def rising_edges(t: list[float], alert: list[bool]) -> list[float]:
    return [t[i] for i in range(len(alert)) if alert[i] and (i == 0 or not alert[i - 1])]


def outcome(t: list[float], alert: list[bool], onset: float | None, edges: list[float] | None = None) -> dict[str, Any]:
    """Detection summary of one vehicle's alert stream. onset None = clean vehicle (every alert is false)."""
    edges = rising_edges(t, alert) if edges is None else edges
    first = next((ti for ti, a in zip(t, alert) if a and onset is not None and ti >= onset), None)
    return {
        "detected": None if onset is None else first is not None,
        "first_alert_t": first,
        "latency_s": None if first is None else first - onset,
        "false_alerts": sum(1 for e in edges if onset is None or e < onset),
        "alert_edges_t": edges,
        "alert_frames": int(sum(alert)),
    }


def free_udp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def parse_vid(v: Any) -> int:
    return int(v, 16) if isinstance(v, str) else int(v)


_NONFINITE = re.compile(r"(?<![\w\"])(-?)(nan|inf)(?![\w\"])", re.IGNORECASE)


def parse_jsonl(path: Path) -> tuple[list[dict], int]:
    """Records of a JSONL file, tolerating bare nan/inf tokens. Returns (records, unparseable line count)."""
    out, bad = [], 0
    if not path.exists():
        return out, bad
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            fixed = _NONFINITE.sub(lambda m: m.group(1) + ("NaN" if m.group(2).lower() == "nan" else "Infinity"), line)
            try:
                out.append(json.loads(fixed))
            except json.JSONDecodeError:
                bad += 1
    return out, bad


# ---------------------------------------------------------------------------------------------- Python side


@dataclass
class PyRun:
    seq: list[int] = field(default_factory=list)
    t: list[float] = field(default_factory=list)
    alert: list[bool] = field(default_factory=list)
    score: list[float] = field(default_factory=list)
    l0_ok: list[bool] = field(default_factory=list)
    latency_us: list[float] = field(default_factory=list)


def run_python(vehicle: udp_replay.Vehicle) -> PyRun:
    """The Python ground side on the exact packets the monitor gets: decode + LinkMonitor + BaseDetector."""
    monitor = LinkMonitor(vehicle.vehicle_id)
    det = make_detector("base", SCHEMAS_BY_NAME[vehicle.schema])
    run = PyRun()
    clock = time.perf_counter_ns
    for packet in vehicle.packets:
        t0 = clock()
        frame, l0 = monitor.receive(packet)
        verdict = det.update(frame, l0)
        run.latency_us.append((clock() - t0) / 1000.0)
        run.seq.append(frame.seq)
        run.t.append(frame.t)
        run.alert.append(bool(verdict.alert))
        run.score.append(float(verdict.score))
        run.l0_ok.append(l0.ok)
    return run


# ---------------------------------------------------------------------------------------------- C++ side


def run_monitor(exe: Path, config: Path, vehicles: list[udp_replay.Vehicle], speed: float, rate_hz: float | None,
                workdir: Path, idle_exit: float, startup_s: float) -> dict[str, Any]:
    """Launch vbw_edge, send every packet, wait for its idle exit. Returns send stats, records and process info."""
    port = free_udp_port()
    jsonl = workdir / f"edge_{port}.jsonl"
    launcher = [sys.executable] if exe.suffix == ".py" else []   # a .py stand-in with the same CLI, for testing
    cmd = [*launcher, str(exe), "--config", str(config), "--port", str(port), "--jsonl", str(jsonl), "--log-frames",
           "--idle-exit", f"{idle_exit:g}"]
    stdout_path, stderr_path = workdir / f"edge_{port}.out", workdir / f"edge_{port}.err"
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err, cwd=str(ROOT))
        time.sleep(startup_s)   # config + ONNX load; there is no ready signal
        if proc.poll() is not None:
            raise SystemExit(f"vbw_edge exited during startup (code {proc.returncode}):\n"
                             f"{stderr_path.read_text(errors='replace')[-2000:]}")
        stats = udp_replay.send(vehicles, "127.0.0.1", port, speed, rate_hz)
        status = "exited"
        try:
            proc.wait(timeout=idle_exit + 60.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            status = "killed (no idle exit within idle_exit + 60 s)"
    records, bad = parse_jsonl(jsonl)
    return {
        "cmd": " ".join(cmd), "port": port, "send": stats, "records": records, "bad_lines": bad,
        "returncode": proc.returncode, "status": status,
        "stderr_tail": stderr_path.read_text(errors="replace")[-1000:],
        "stdout_tail": stdout_path.read_text(errors="replace")[-1000:],
    }


def agreement(py: PyRun, frames: list[dict], cpp_edges: list[float], py_edges: list[float]) -> dict[str, Any]:
    """Frame-by-frame join on (seq, occurrence): alert flags and scores, plus rising-edge times."""
    by_seq: dict[int, deque[int]] = defaultdict(deque)
    for i, s in enumerate(py.seq):
        by_seq[s].append(i)
    compared, mismatches, unmatched, max_diff = 0, 0, 0, 0.0
    for rec in frames:
        queue = by_seq.get(int(rec.get("seq", -1)))
        i = queue.popleft() if queue else None
        if i is None or abs(float(rec.get("t", math.nan)) - py.t[i]) > T_MATCH:
            unmatched += 1
            continue
        compared += 1
        mismatches += int(bool(rec.get("alert")) != py.alert[i])
        score = rec.get("score")
        if isinstance(score, (int, float)) and math.isfinite(score):
            max_diff = max(max_diff, abs(float(score) - py.score[i]))
    equal = len(cpp_edges) == len(py_edges) and all(abs(a - b) <= T_MATCH for a, b in zip(cpp_edges, py_edges))
    return {"alert_edges_equal": equal, "first_alert_equal": (cpp_edges[:1] == py_edges[:1]) if not equal else True,
            "frames_compared": compared, "frames_unmatched": unmatched, "alert_flag_mismatches": mismatches,
            "max_score_diff": max_diff if compared else None}


# ---------------------------------------------------------------------------------------------- cases


class Cache:
    """Vehicles by full identity; Python runs by stream identity (vehicle_id doesn't change the detector's input)."""

    def __init__(self, max_frames: int | None):
        self.max_frames = max_frames
        self.vehicles: dict[tuple, udp_replay.Vehicle] = {}
        self.python: dict[tuple, PyRun] = {}

    def fleet(self, case: dict[str, Any]) -> list[udp_replay.Vehicle]:
        out = []
        for i, (scenario, vid) in enumerate(udp_replay.fleet_ids([case["scenario"]], case["vehicles"])):
            attacked = case["attacked"] is None or i < case["attacked"]
            attack = case["attack"] if attacked else "none"
            key = (scenario, case["seed"] + i, vid, attack, case["attack_start"], case["intensity"], self.max_frames)
            if key not in self.vehicles:
                t0 = time.perf_counter()
                self.vehicles[key] = udp_replay.build_vehicle(scenario, case["seed"] + i, vid, attack,
                                                              case["attack_start"], case["intensity"], self.max_frames)
                v = self.vehicles[key]
                print(f"  built {v.vehicle_id} (vid 0x{v.vid_hash:04x}, seed {v.seed}, {attack}): "
                      f"{len(v.packets)} frames, {time.perf_counter() - t0:.1f}s")
            out.append(self.vehicles[key])
        return out

    def run_python(self, v: udp_replay.Vehicle) -> PyRun:
        key = (v.scenario, v.seed, v.attack, json.dumps(v.attack_params, sort_keys=True), len(v.packets))
        if key not in self.python:
            t0 = time.perf_counter()
            self.python[key] = run_python(v)
            print(f"  python {v.vehicle_id}: {len(v.packets)} frames, {time.perf_counter() - t0:.1f}s")
        return self.python[key]


def onset_of(v: udp_replay.Vehicle) -> float | None:
    if v.attack == "none":
        return None
    return v.onset_t if v.onset_t is not None else v.attack_start


def evaluate_case(name: str, case: dict[str, Any], cache: Cache, args: argparse.Namespace,
                  workdir: Path) -> dict[str, Any]:
    attacked = "all" if case["attacked"] is None else case["attacked"]
    pace = f"{case['rate_hz']:g} Hz" if case["rate_hz"] else "frame t"
    print(f"[{name}] {case['vehicles']} vehicle(s), attack {case['attack']} on {attacked}, "
          f"speed {case['speed']:g} ({'max rate' if case['speed'] == 0 else pace})")
    vehicles = cache.fleet(case)
    py_runs = [cache.run_python(v) for v in vehicles]
    result: dict[str, Any] = {"case": {k: case[k] for k in CASE_DEFAULTS}, "vehicles": []}

    cpp: dict[str, Any] | None = None
    by_vid_frames: dict[int, list[dict]] = defaultdict(list)
    by_vid_alerts: dict[int, list[dict]] = defaultdict(list)
    summary: dict[str, Any] | None = None
    if not args.python_only:
        cpp = run_monitor(args.exe, args.config, vehicles, case["speed"], case["rate_hz"], workdir,
                          args.idle_exit, args.startup_s)
        for rec in cpp["records"]:
            kind = rec.get("type")
            if kind == "frame" and "vid" in rec:
                by_vid_frames[parse_vid(rec["vid"])].append(rec)
            elif kind == "alert" and "vid" in rec:
                by_vid_alerts[parse_vid(rec["vid"])].append(rec)
            elif kind == "summary":
                summary = rec

    for v, py in zip(vehicles, py_runs):
        onset = onset_of(v)
        py_out = outcome(py.t, py.alert, onset)
        entry: dict[str, Any] = {"vehicle_id": v.vehicle_id, "vid_hash": f"0x{v.vid_hash:04x}", "attack": v.attack,
                                 "onset_t": onset, "frames_sent": len(v.packets), "python": py_out}
        if cpp is not None:
            frames = by_vid_frames.get(v.vid_hash, [])
            alerts = by_vid_alerts.get(v.vid_hash, [])
            ft = [float(r.get("t", math.nan)) for r in frames]
            fa = [bool(r.get("alert")) for r in frames]
            edges = [float(r["t"]) for r in alerts if "t" in r] if alerts or not frames else rising_edges(ft, fa)
            entry["cpp"] = {**outcome(ft, fa, onset, edges), "frames_received": len(frames),
                            "frames_dropped": len(v.packets) - len(frames)}
            entry["agreement"] = agreement(py, frames, edges, py_out["alert_edges_t"])
        result["vehicles"].append(entry)

    all_py_lat = [x for py in py_runs for x in py.latency_us]
    py_lat = percentiles(all_py_lat)
    result["python"] = {"latency_us": py_lat,
                        "compute_frames_per_s": 1e6 / py_lat["mean"] if py_lat["mean"] else None}
    if cpp is not None:
        lat = [float(r["latency_us"]) for fs in by_vid_frames.values() for r in fs
               if isinstance(r.get("latency_us"), (int, float))]
        lat_s = percentiles(lat)
        sent = cpp["send"]["frames_sent"]
        received = sum(len(fs) for fs in by_vid_frames.values())
        result["cpp"] = {
            "latency_us": lat_s,
            "frames_per_s": (summary or {}).get("frames_per_s"),
            "compute_frames_per_s": 1e6 / lat_s["mean"] if lat_s["mean"] else None,
            "frames_sent": sent,
            "frames_received": received,
            "frames_dropped": sent - received,
            "summary": summary,
            "jsonl_bad_lines": cpp["bad_lines"],
            "process": {k: cpp[k] for k in ("cmd", "returncode", "status")},
        }
        if cpp["returncode"] not in (0, None) or cpp["status"] != "exited":
            result["cpp"]["process"]["stderr_tail"] = cpp["stderr_tail"]
        result["send"] = cpp["send"]
    result["summary"] = case_summary(result)
    return result


def case_summary(result: dict[str, Any]) -> dict[str, Any]:
    vs = result["vehicles"]
    attacked = [v for v in vs if v["onset_t"] is not None]
    out: dict[str, Any] = {"vehicles": len(vs), "attacked": len(attacked)}
    for side in ("python", "cpp"):
        if not vs or side not in vs[0]:
            continue
        lat = [v[side]["latency_s"] for v in attacked if v[side]["latency_s"] is not None]
        out[side] = {
            "detected": sum(bool(v[side]["detected"]) for v in attacked),
            "mean_latency_s": float(np.mean(lat)) if lat else None,
            "false_alerts": sum(v[side]["false_alerts"] for v in vs),
        }
    if vs and "agreement" in vs[0]:
        out["agreement"] = {
            "all_alert_edges_equal": all(v["agreement"]["alert_edges_equal"] for v in vs),
            "alert_flag_mismatches": sum(v["agreement"]["alert_flag_mismatches"] for v in vs),
            "frames_compared": sum(v["agreement"]["frames_compared"] for v in vs),
            "max_score_diff": max((v["agreement"]["max_score_diff"] or 0.0) for v in vs),
        }
    return out


def print_case(name: str, r: dict[str, Any]) -> None:
    s = r["summary"]
    py = s.get("python", {})
    line = (f"[{name}] python: {py.get('detected')}/{s['attacked']} detected, latency {fmt(py.get('mean_latency_s'))} s, "
            f"{py.get('false_alerts')} false alerts, {fmt(r['python']['latency_us']['p50'])} us/frame p50")
    print(line)
    if "cpp" in r:
        c, cs, ag = r["cpp"], s["cpp"], s.get("agreement", {})
        print(f"[{name}] c++:    {cs['detected']}/{s['attacked']} detected, latency {fmt(cs['mean_latency_s'])} s, "
              f"{cs['false_alerts']} false alerts, latency p50/p99/max {fmt(c['latency_us']['p50'])}/"
              f"{fmt(c['latency_us']['p99'])}/{fmt(c['latency_us']['max'])} us, {fmt(c['frames_per_s'])} frames/s, "
              f"{c['frames_dropped']} dropped of {c['frames_sent']}")
        print(f"[{name}] agree:  edges equal {ag.get('all_alert_edges_equal')}, "
              f"{ag.get('alert_flag_mismatches')} alert-flag mismatches over {ag.get('frames_compared')} frames, "
              f"max score diff {fmt(ag.get('max_score_diff'), 2)}")


def fmt(x: Any, digits: int = 1) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{digits}e}" if digits == 2 else f"{x:.{digits}f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=",".join(CASES), help="comma list of case names (see --list)")
    ap.add_argument("--list", action="store_true", help="print the cases and exit")
    ap.add_argument("--python-only", action="store_true", help="skip the C++ monitor (no exe needed)")
    ap.add_argument("--exe", type=Path, default=EXE)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--max-frames", type=int, default=None, help="truncate every flight (smoke tests)")
    ap.add_argument("--idle-exit", type=float, default=3.0)
    ap.add_argument("--startup-s", type=float, default=1.5, help="wait this long after launching the exe")
    ap.add_argument("--workdir", type=Path, default=None, help="keep the monitor's JSONL/stdout here")
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/edge_eval.json (results/edge_eval_python_only.json with --python-only)")
    args = ap.parse_args()

    if args.list:
        for name, overrides in CASES.items():
            print(f"{name:12s} {overrides}")
        return
    names = [c for c in args.cases.split(",") if c]
    unknown = [c for c in names if c not in CASES]
    if unknown:
        ap.error(f"unknown case(s) {unknown}; have {list(CASES)}")
    if not args.python_only:
        if not args.exe.exists():
            raise SystemExit(f"C++ monitor not found at {args.exe}. Build it first (edge/build.ps1), pass --exe, or "
                             f"run with --python-only.")
        if not args.config.exists():
            raise SystemExit(f"config {args.config} not found; run scripts/export_edge_config.py --schema falcon9")
    out_path = args.out or RESULTS / ("edge_eval_python_only.json" if args.python_only else "edge_eval.json")

    cache = Cache(args.max_frames)
    results: dict[str, Any] = {}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="vbw_edge_eval_") as tmp:
        workdir = args.workdir or Path(tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        for name in names:
            case = {**CASE_DEFAULTS, **CASES[name]}
            results[name] = evaluate_case(name, case, cache, args, workdir)
            print_case(name, results[name])

    doc = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_only": args.python_only,
        "exe": None if args.python_only else str(args.exe.relative_to(ROOT) if args.exe.is_relative_to(ROOT)
                                                  else args.exe),
        "config": str(args.config.relative_to(ROOT) if args.config.is_relative_to(ROOT) else args.config),
        "max_frames": args.max_frames,
        "notes": {
            "latency_s": "mission seconds from attack onset (first forged frame on the wire) to the first alert frame",
            "false_alerts": "alert rising edges before onset, or any on a clean vehicle",
            "cpp_latency_us": "vbw_edge per-frame recvfrom -> verdict (frame lines); python: decode + LinkMonitor + "
                              "BaseDetector.update in-process",
            "frames_dropped": "frames sent minus frame lines the monitor logged for that vehicle",
            "agreement": "frames joined on (seq, occurrence); alert edges equal within 1e-3 s",
        },
        "elapsed_s": round(time.time() - t0, 1),
        "cases": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
