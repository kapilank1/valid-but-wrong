"""Download NASA SMAP/MSL telemetry anomaly data (Hundman et al., KDD 2018).

    uv run python scripts/fetch_smap.py

The original S3 bucket (telemanom) now returns 403, so this pulls the identical .npy files mirrored
in the TranAD repo. Lands in data/raw/smap_msl/{train,test}/<chan>.npy + labeled_anomalies.csv.
Load with np.load(path, allow_pickle=False).
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "smap_msl"
API = "https://api.github.com/repos/imperial-qore/TranAD/contents/data/SMAP_MSL"
LABELS = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"


def get(url: str, attempts: int = 4) -> bytes:
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except OSError:
            if i == attempts - 1:
                raise
    raise AssertionError("unreachable")


def download(item: tuple[str, Path]) -> None:
    url, dest = item
    if not dest.exists():
        dest.write_bytes(get(url))


def main() -> None:
    jobs = []
    for split in ("train", "test"):
        (OUT / split).mkdir(parents=True, exist_ok=True)
        for entry in json.loads(get(f"{API}/{split}")):
            if entry["name"].endswith(".npy"):
                jobs.append((entry["download_url"], OUT / split / entry["name"]))
    (OUT / "labeled_anomalies.csv").write_bytes(get(LABELS))
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(download, jobs))
    print(f"downloaded {len(jobs)} files to {OUT}")


if __name__ == "__main__":
    main()
