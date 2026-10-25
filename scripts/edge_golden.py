"""Golden parity files for the C++ edge monitor: the Python BaseDetector's per-frame output (edge/SPEC.md section 3).

    uv run python scripts/edge_golden.py                 # all sets -> edge/testdata/golden_<name>.csv + README.md
    uv run python scripts/edge_golden.py --only ses9_bias

Every frame goes through the same link as the live pipeline and udp_replay.py (source -> encode/decode -> attack ->
encode with a valid CRC), then the ground side decodes it with a LinkMonitor and scores it with the Python
BaseDetector. Each row records the inputs (t, seq, the float32-rounded values the detector saw, and the exact UDP
payload as `packet_hex`) and every intermediate the C++ must reproduce: L0 flags, whether the detector accepted the
frame into its history, signed L1 ratios per rule, L2 z per channel, the three scores and the alert state.

The C++ parity test feeds `packet_hex` through decode + LinkMonitor + BaseDetector and compares column by column, so
it needs no Python and no re-encoding (which matters for the link-fault set, whose packets are deliberately broken).
Numbers are repr() (exact round trip through strtod), NaN is `nan`, booleans are 0/1.
"""

from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import udp_replay  # noqa: E402

from vbw.detector.physics import RULES_BY_SCHEMA  # noqa: E402
from vbw.detector.registry import make_detector  # noqa: E402
from vbw.link.codec import LinkMonitor, crc16_ccitt  # noqa: E402
from vbw.schema import FALCON9, ChannelSchema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TESTDATA = ROOT / "edge" / "testdata"
SCENARIO = "falcon9:ses-9"
SEED = 0
ATTACK_START = 190.0

# Byte offsets in a FALCON9 packet (vbw/link/codec.py): sync 0-3, version 4, schema 5, vid 6-7, seq 8-9, t 10-17,
# n 18, values 19 + 4*i, crc last 2.
OFF_SEQ, OFF_T, OFF_VALUES = 8, 10, 19
FAULT_FRAMES = 420


def recrc(packet: bytearray) -> bytes:
    """Recompute the CRC so only the deliberately broken field fails L0."""
    packet[-2:] = struct.pack(">H", crc16_ccitt(bytes(packet[:-2])))
    return bytes(packet)


def set_value(packet: bytes, channel: int, value: float) -> bytes:
    p = bytearray(packet)
    struct.pack_into(">f", p, OFF_VALUES + 4 * channel, value)
    return recrc(p)


def link_fault_packets(packets: list[bytes], t: list[float]) -> tuple[list[bytes], list[tuple[int, str]]]:
    """A clean stream with one of each link fault injected. Returns the packets and (row, description) notes."""
    ch = {c: i for i, c in enumerate(FALCON9.channels)}
    out: list[bytes] = []
    notes: list[tuple[int, str]] = []

    def note(text: str) -> None:
        notes.append((len(out) - 1, text))

    for i, packet in enumerate(packets):
        if 200 <= i < 205:
            if i == 200:
                notes.append((len(out), "frames 200-204 dropped: seq jump 199->205 (seq_ok=0, still accepted)"))
            continue
        if i == 130:
            p = bytearray(packet)
            p[OFF_VALUES + 4 * ch["altitude"] + 3] ^= 0x5A          # one flipped byte in the altitude value
            out.append(bytes(p))
            note("bad CRC: altitude byte corrupted, CRC left as sent (crc_ok=0, rejected)")
        elif i == 160:
            p = bytearray(packet)
            p[0] = 0x1B                                              # 0x1BCFFC1D
            out.append(recrc(p))
            note("bad sync marker 0x1BCFFC1D with a valid CRC (sync_ok=0, rejected)")
        elif i == 240:
            p = bytearray(packet)
            struct.pack_into(">d", p, OFF_T, t[i - 1] - 0.05)       # older than the previous frame
            out.append(recrc(p))
            note("non-monotonic t (previous t - 0.05) with a valid CRC (time_ok=0, rejected)")
        elif i == 280:
            out.append(packet)
            out.append(packet)                                       # exact duplicate (replayed packet)
            note("duplicate of the previous packet (seq_ok=0, time_ok=0, rejected)")
        elif i == 300:
            p = bytearray(packet)
            struct.pack_into(">H", p, OFF_SEQ, (struct.unpack_from(">H", p, OFF_SEQ)[0] + 1000) & 0xFFFF)
            out.append(recrc(p))
            note("seq rewritten +1000 with a valid CRC (seq_ok=0 here and on the next frame, both accepted)")
        elif i == 320:
            out.append(set_value(packet, ch["vz"], float("nan")))
            note("NaN vz with a valid CRC (L0 ok, rejected as non-finite)")
        elif i == 360:
            out.append(set_value(packet, ch["accel"], float("inf")))
            note("+inf accel with a valid CRC (L0 ok, rejected as non-finite)")
        else:
            out.append(packet)
    return out, notes


def columns(schema: ChannelSchema) -> list[str]:
    rules = [r.name for r in RULES_BY_SCHEMA[schema.name]]
    return (["t", "seq", *schema.channels, "l0_ok", "accepted", *(f"l1r.{r}" for r in rules), "l1_score",
             *(f"l2z.{c}" for c in schema.channels), "l2_score", "score", "alert",
             "sync_ok", "crc_ok", "seq_ok", "time_ok", "packet_hex"])


def num(x: float) -> str:
    return repr(float(x))


def b(flag: bool) -> str:
    return "1" if flag else "0"


def score_packets(packets: list[bytes], vehicle_id: str, schema: ChannelSchema = FALCON9) -> list[list[str]]:
    """Python ground side, frame by frame: decode + LinkMonitor + BaseDetector. One CSV row per packet."""
    rules = [r.name for r in RULES_BY_SCHEMA[schema.name]]
    zeros_l1, zeros_l2 = ["0.0"] * len(rules), ["0.0"] * len(schema.channels)
    monitor = LinkMonitor(vehicle_id)
    det = make_detector("base", schema)
    rows = []
    for packet in packets:
        frame, l0 = monitor.receive(packet)
        verdict = det.update(frame, l0)
        accepted = det.last_accepted
        if accepted:
            l1r = [num(det.last_l1.ratios[r]) for r in rules]
            l2z = [num(z) for z in np.asarray(det.last_z, dtype=float)]
        else:  # last_l1 / last_z still hold the previous accepted frame; the frame's own scores are all 0
            l1r, l2z = zeros_l1, zeros_l2
        rows.append([num(frame.t), str(frame.seq), *(num(frame.channels[c]) for c in schema.channels),
                     b(l0.ok), b(accepted), *l1r, num(verdict.l1_score), *l2z, num(verdict.l2_score),
                     num(verdict.score), b(verdict.alert),
                     b(l0.sync_ok), b(l0.crc_ok), b(l0.seq_ok), b(l0.time_ok), packet.hex()])
    return rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def build_set(name: str) -> tuple[udp_replay.Vehicle, list[bytes], list[tuple[int, str]]]:
    if name == "ses9_clean":
        v = udp_replay.build_vehicle(SCENARIO, SEED)
    elif name == "ses9_bias":
        v = udp_replay.build_vehicle(SCENARIO, SEED, attack="bias", attack_start=ATTACK_START, intensity=1.0)
    elif name == "ses9_ppo":
        v = udp_replay.build_vehicle(SCENARIO, SEED, attack="ppo", attack_start=ATTACK_START)
    elif name == "link_faults":
        v = udp_replay.build_vehicle(SCENARIO, SEED, max_frames=FAULT_FRAMES)
        packets, notes = link_fault_packets(v.packets, v.t)
        return v, packets, notes
    else:
        raise KeyError(name)
    return v, v.packets, []


SETS: dict[str, str] = {
    "ses9_clean": f"{SCENARIO}, seed {SEED}, no attack (full flight).",
    "ses9_bias": f"{SCENARIO}, seed {SEED}, bias on altitude, intensity 1.0 (+3000 m) from T+{ATTACK_START:g} s.",
    "ses9_ppo": (f"{SCENARIO}, seed {SEED}, PPO round-1 policy (models/ppo_falcon9.zip, base shadow detector, torch "
                 f"1 thread) from T+{ATTACK_START:g} s."),
    "link_faults": f"first {FAULT_FRAMES} frames of {SCENARIO} seed {SEED}, clean, with injected link faults (below).",
}


def write_readme(summary: dict[str, dict], notes: list[tuple[int, str]], header: list[str]) -> None:
    lines = [
        "# vbw_edge golden parity files",
        "",
        "Generated by `scripts/edge_golden.py` (do not edit by hand). Each `golden_<name>.csv` is the Python",
        "`BaseDetector` (`vbw/detector/base.py`) run frame by frame on the exact UDP payloads in its `packet_hex`",
        "column: decode + `LinkMonitor` + `BaseDetector.update`, as the ground station does. The C++ parity tests feed",
        "`packet_hex` through `decode` + `LinkMonitor` + `BaseDetector` and compare every other column.",
        "",
        "Regenerate (after recalibrating L1/L2/alerting, also re-run `scripts/export_edge_config.py --schema all`):",
        "",
        "```",
        "uv run python scripts/export_edge_config.py --schema all",
        "uv run python scripts/edge_golden.py",
        "```",
        "",
        "## Files",
        "",
        "| File | Rows | Alert frames | Rejected frames | Contents |",
        "|---|---|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(f"| `golden_{name}.csv` | {s['rows']} | {s['alert_frames']} | {s['rejected']} | {SETS[name]} |")
    lines += [
        "",
        "## Columns",
        "",
        "One header row, then one row per received packet, in arrival order:",
        "",
        "```",
        ", ".join(header),
        "```",
        "",
        "- `t`, `seq`, channel values: the decoded frame (values are float32 on the wire, printed as the double the",
        "  detector works in).",
        "- `l0_ok`: all four L0 checks passed; `sync_ok`, `crc_ok`, `seq_ok`, `time_ok`: the individual checks",
        "  (`LinkMonitor` updates its last seq / t on every decodable packet, including ones that fail).",
        "- `accepted`: `BaseDetector.last_accepted`, i.e. the frame entered the detector's history (sync and CRC ok,",
        "  all values finite, t > last accepted t). For a rejected frame every `l1r.*`, `l2z.*` and score is 0, and",
        "  `alert` is still the alert state machine fed a score of 0.",
        "- `l1r.<rule>`: signed residual / threshold of the newest frame (`L1Result.ratios`), 0 while not evaluable.",
        "- `l2z.<channel>`: `BaseDetector.last_z` (0 while the history is shorter than `l2.window + 2`).",
        "- `l1_score`, `l2_score`, `score`, `alert`: the `Verdict` fields.",
        "- `packet_hex`: the exact UDP payload (lowercase hex).",
        "",
        "Numbers are Python `repr()` (round-trip exact through `strtod`), NaN is `nan`, booleans are `0`/`1`.",
        "Parity tolerances (edge/SPEC.md): L1 ratios abs 1e-9 (relative 1e-9 for large values), L2 z abs 1e-4,",
        "scores abs 1e-6, alerts exact.",
        "",
        "## Link faults (`golden_link_faults.csv`)",
        "",
        "Row numbers are 0-based data rows (row 0 is the first line after the header).",
        "",
    ]
    lines += [f"- row {row}: {text}" for row, text in notes]
    (TESTDATA / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(SETS), help="comma list of sets")
    args = ap.parse_args()
    wanted = [s for s in args.only.split(",") if s]
    header = columns(FALCON9)
    summary: dict[str, dict] = {}
    notes: list[tuple[int, str]] = []
    for name in wanted:
        t0 = time.perf_counter()
        vehicle, packets, set_notes = build_set(name)
        rows = score_packets(packets, vehicle.vehicle_id)
        path = TESTDATA / f"golden_{name}.csv"
        write_csv(path, header, rows)
        i_alert, i_acc = header.index("alert"), header.index("accepted")
        summary[name] = {"rows": len(rows), "alert_frames": sum(r[i_alert] == "1" for r in rows),
                         "rejected": sum(r[i_acc] == "0" for r in rows)}
        notes = set_notes or notes
        first = next((r[0] for r in rows if r[i_alert] == "1"), None)
        print(f"{name}: {len(rows)} rows, {summary[name]['alert_frames']} alert frames (first t={first}), "
              f"{summary[name]['rejected']} rejected, {path.stat().st_size / 1e6:.2f} MB, "
              f"{time.perf_counter() - t0:.1f}s")
    if set(wanted) == set(SETS):
        write_readme(summary, notes, header)
        print(f"README -> {TESTDATA / 'README.md'}")


if __name__ == "__main__":
    main()
