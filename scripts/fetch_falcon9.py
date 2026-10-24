"""Download real Falcon 9 webcast telemetry (shahar603/Telemetry-Data) and resample to 10 Hz CSVs.

    uv run python scripts/fetch_falcon9.py

Raw JSON lands in data/raw/falcon9/ (gitignored). Processed CSVs + index.json land in
data/processed/falcon9/ (committed, so the demo works from a fresh clone).
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "falcon9"
OUT = ROOT / "data" / "processed" / "falcon9"
INDEX_URL = "https://raw.githubusercontent.com/shahar603/Telemetry-Data/master/Laucnhes.json"
RATE_HZ = 10.0

# analysed.json field -> (our channel, scale to SI)
FIELDS = {
    "velocity": ("speed", 1.0),
    "altitude": ("altitude", 1000.0),
    "velocity_y": ("vz", 1.0),
    "velocity_x": ("vx", 1.0),
    "downrange_distance": ("downrange", 1000.0),
    "acceleration": ("accel", 1.0),
    "angle": ("angle", 1.0),
    "q": ("q", 1.0),
}


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def fetch_json(url: str, dest: Path):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=30) as r:
            dest.write_bytes(r.read())
    return json.loads(dest.read_text())


def resample(analysed: dict) -> pd.DataFrame:
    t = np.asarray(analysed["time"], dtype=float)
    # drop duplicate timestamps, keep order
    t, keep = np.unique(t, return_index=True)
    t_new = np.arange(t[0], t[-1], 1.0 / RATE_HZ)
    out = {"t": np.round(t_new, 3)}
    for src, (dst, scale) in FIELDS.items():
        y = np.asarray(analysed[src], dtype=float)[keep] * scale
        out[dst] = PchipInterpolator(t, y)(t_new)
    return pd.DataFrame(out)


def main() -> None:
    launches = fetch_json(INDEX_URL, RAW / "launches.json")
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for launch in launches:
        urls = launch.get("JSON") or {}
        if not urls.get("analysed"):
            continue
        slug = slugify(launch["mission_name"])
        try:
            analysed = fetch_json(urls["analysed"], RAW / slug / "analysed.json")
            events = fetch_json(urls["events"], RAW / slug / "events.json") if urls.get("events") else {}
        except Exception as exc:  # network hiccup on one launch shouldn't kill the run
            print(f"skip {slug}: {exc}")
            continue
        df = resample(analysed)
        df.to_csv(OUT / f"{slug}.csv", index=False, float_format="%.4f")
        index.append({
            "slug": slug,
            "mission_name": launch["mission_name"],
            "flight_number": launch.get("flight_number"),
            "stage": launch.get("analysed_stage"),
            "duration_s": float(df["t"].iloc[-1]),
            "rate_hz": RATE_HZ,
            "events": {k: v for k, v in (events or {}).items() if v is not None},
        })
        print(f"{slug}: {len(df)} frames, stage {launch.get('analysed_stage')}")
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {len(index)} launches to {OUT}")


if __name__ == "__main__":
    main()
