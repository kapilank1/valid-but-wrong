"""Launch the ground-station pipeline server.

    uv run python scripts/serve.py                 # http://127.0.0.1:8000, WebSocket at /ws
    uv run python scripts/serve.py --host 0.0.0.0 --port 8000 --reload

Loads .env from the repo root if present (DATABASE_URL turns on the Tiger Data store).
Equivalent to `uv run uvicorn vbw.pipeline.server:app --port 8000`.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes (dev only)")
    parser.add_argument("--env-file", default=str(ROOT / ".env"))
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    env_file = args.env_file if Path(args.env_file).is_file() else None
    uvicorn.run(
        "vbw.pipeline.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT / "vbw")] if args.reload else None,
        env_file=env_file,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
