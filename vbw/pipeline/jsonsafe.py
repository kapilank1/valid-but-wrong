"""Strict JSON for the wire: NaN/inf become null, numpy scalars become Python numbers.

Browsers reject the bare `NaN` token that Python's json module emits by default, so every message the
pipeline sends (WebSocket and REST) goes through `dumps`.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID


def _default(obj: Any) -> Any:
    """Fallback for types json can't encode: numpy scalars/arrays, DB values, sets, dataclass-likes."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, UUID):
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        try:
            return obj.item()
        except (TypeError, ValueError):
            pass
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return str(obj)


def sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with None and coerce unknown types to JSON-able ones."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [sanitize(v) for v in obj]
    return sanitize(_default(obj))


def dumps(obj: Any) -> str:
    """Compact strict JSON. Fast path when the payload is already clean, sanitizing fallback otherwise."""
    try:
        return json.dumps(obj, allow_nan=False, separators=(",", ":"), default=_default)
    except ValueError:
        return json.dumps(sanitize(obj), allow_nan=False, separators=(",", ":"))
