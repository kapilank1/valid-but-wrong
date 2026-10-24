"""Detector registry (docs/CONTRACT.md): list_detectors() and make_detector(name, schema)."""

from __future__ import annotations

from vbw.schema import ChannelSchema, Detector

DETECTORS = [
    {"name": "base", "label": "Physics + learned (L1+L2)",
     "description": "Physics-consistency rules plus a GRU next-step predictor, with persistent alerting."},
    {"name": "hardened", "label": "Hardened (L1+L2+L3)",
     "description": "Base detector plus a discriminator trained on attacker-forged windows (falls back to base if untrained)."},
]


def list_detectors() -> list[dict]:
    return [dict(d) for d in DETECTORS]


def make_detector(name: str, schema: ChannelSchema) -> Detector:
    if name not in {d["name"] for d in DETECTORS}:
        raise KeyError(f"unknown detector {name!r}; expected one of {[d['name'] for d in DETECTORS]}")
    if schema.name == "spacecraft":
        from vbw.detector.spacecraft import SpacecraftDetector

        detector = SpacecraftDetector(schema)
        detector.name = name
        return detector
    if name == "hardened":
        from vbw.detector.l3 import HardenedDetector

        return HardenedDetector(schema)
    from vbw.detector.base import BaseDetector

    return BaseDetector(schema)
