"""Builders for the WebSocket messages in docs/CONTRACT.md. Plain dicts; `jsonsafe.dumps` serializes them."""

from __future__ import annotations

from typing import Any

from vbw.schema import Frame, Verdict


def hello(catalog: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    return {"type": "hello", "catalog": catalog, "status": status}


def scenario(info: dict[str, Any]) -> dict[str, Any]:
    return {"type": "scenario", "scenario": info}


def status(status_dict: dict[str, Any]) -> dict[str, Any]:
    return {"type": "status", "status": status_dict}


def tick(truth: Frame, reported: Frame, verdict: Verdict, attack: str, active: bool) -> dict[str, Any]:
    return {
        "type": "tick",
        "truth": truth.to_dict(),
        "reported": reported.to_dict(),
        "verdict": verdict.to_dict(),
        "attack": {"name": attack, "active": active},
    }


def alert(verdict: Verdict, spoken: str) -> dict[str, Any]:
    return {"type": "alert", "verdict": verdict.to_dict(), "spoken": spoken}


# Message types a slow client may lose (only the newest matter); everything else is always delivered.
DROPPABLE_TYPES = frozenset({"tick"})
