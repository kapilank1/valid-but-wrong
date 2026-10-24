"""Dependency-injection seam between the pipeline and the source/attacker/detector registries.

The pipeline never imports those modules at import time. `default_registries()` binds lazily to
`vbw.sources.registry`, `vbw.attacker.registry` and `vbw.detector.registry` (per docs/CONTRACT.md);
tests pass fakes instead.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Callable

from vbw.schema import Attack, ChannelSchema, Detector, Source

log = logging.getLogger("vbw.pipeline")


@dataclass(frozen=True)
class Registries:
    list_scenarios: Callable[[], list[dict]]
    make_source: Callable[..., Source]                     # (scenario_id, seed=None)
    list_attacks: Callable[[], list[dict]]
    make_attack: Callable[..., Attack]                     # (name, schema, **params)
    list_detectors: Callable[[], list[dict]]
    make_detector: Callable[[str, ChannelSchema], Detector]


def _lazy(module: str, attr: str) -> Callable[..., Any]:
    def call(*args: Any, **kwargs: Any) -> Any:
        return getattr(importlib.import_module(module), attr)(*args, **kwargs)

    call.__name__ = attr
    return call


def default_registries() -> Registries:
    return Registries(
        list_scenarios=_lazy("vbw.sources.registry", "list_scenarios"),
        make_source=_lazy("vbw.sources.registry", "make_source"),
        list_attacks=_lazy("vbw.attacker.registry", "list_attacks"),
        make_attack=_lazy("vbw.attacker.registry", "make_attack"),
        list_detectors=_lazy("vbw.detector.registry", "list_detectors"),
        make_detector=_lazy("vbw.detector.registry", "make_detector"),
    )


def safe_list(fn: Callable[[], list[dict]], what: str) -> list[dict]:
    """Call a list_* registry function; a missing or broken registry yields [] instead of a 500."""
    try:
        return list(fn())
    except Exception as exc:  # noqa: BLE001 - another team's module may not exist yet
        log.warning("catalog: %s unavailable (%s: %s)", what, type(exc).__name__, exc)
        return []


def catalog(registries: Registries) -> dict[str, list[dict]]:
    return {
        "scenarios": safe_list(registries.list_scenarios, "scenarios"),
        "attacks": safe_list(registries.list_attacks, "attacks"),
        "detectors": safe_list(registries.list_detectors, "detectors"),
    }
