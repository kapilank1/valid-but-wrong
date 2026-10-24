"""Store interface used by the pipeline, plus the no-op store used when no database is configured.

The runner calls `record_*` synchronously on every frame, so implementations must return immediately
(queue and write in the background). Query methods are async and return JSON-able dicts; without a
database they return `{"available": False}`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

from vbw.schema import Frame, Verdict


@dataclass
class RunRecord:
    run_id: str
    started_at: datetime                 # wall clock (UTC) when the first frame was played
    scenario: str
    vehicle_id: str
    schema: str
    attack: str
    attack_params: dict[str, Any] = field(default_factory=dict)
    detector: str = "base"
    speed: float = 1.0
    seed: int | None = None

    def time_at(self, t: float) -> datetime:
        """Timestamp stored for a frame: run wall-clock start + mission time."""
        return self.started_at + timedelta(seconds=t)


@dataclass
class TickRecord:
    run: RunRecord
    truth: Frame
    reported: Frame
    verdict: Verdict
    attack: str
    attack_active: bool


class Store(Protocol):
    available: bool

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def record_run_start(self, run: RunRecord) -> None: ...

    def record_tick(self, tick: TickRecord) -> None: ...

    def record_run_end(self, run_id: str, ended_at: datetime, frames: int, alerts: int) -> None: ...

    def stats(self) -> dict[str, Any]: ...

    async def list_runs(self, limit: int = 50) -> dict[str, Any]: ...

    async def timeline(self, run_id: str) -> dict[str, Any]: ...

    async def analytics(self) -> dict[str, Any]: ...


UNAVAILABLE: dict[str, Any] = {"available": False}


class NullStore:
    """Used when DATABASE_URL is unset: accepts everything, stores nothing."""

    available = False

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def record_run_start(self, run: RunRecord) -> None:
        return None

    def record_tick(self, tick: TickRecord) -> None:
        return None

    def record_run_end(self, run_id: str, ended_at: datetime, frames: int, alerts: int) -> None:
        return None

    def stats(self) -> dict[str, Any]:
        return {"available": False}

    async def list_runs(self, limit: int = 50) -> dict[str, Any]:
        return dict(UNAVAILABLE)

    async def timeline(self, run_id: str) -> dict[str, Any]:
        return dict(UNAVAILABLE)

    async def analytics(self) -> dict[str, Any]:
        return dict(UNAVAILABLE)
