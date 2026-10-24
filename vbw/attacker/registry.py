"""Attack registry: the single place the pipeline looks up attacks by name.

``list_attacks()`` describes every attack (name, label, which schemas it fits, and its tunable params
with defaults) so the console can render controls. ``make_attack(name, schema, **params)`` builds one.
``"none"`` (identity) and ``"ppo"`` (the learned attacker) always appear; ppo reports
``available: false`` when no trained policy is on disk for the schema.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vbw.schema import ChannelSchema
from vbw.attacker.scripted import SCRIPTED, _Attack

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

# Common params on every attack. Individual attacks add their own below.
_COMMON = {"start_t": 0.0, "intensity": 0.5}

# name -> (label, description, schemas it applies to, extra params with defaults)
_META: dict[str, tuple[str, str, list[str], dict[str, Any]]] = {
    "none": ("No attack", "Identity passthrough; reported == truth.",
             ["falcon9", "quad", "spacecraft"], {}),
    "bias": ("Constant bias", "Fixed offset added to the target channel(s). L1 catches it fast.",
             ["falcon9", "quad", "spacecraft"], {"channels": [], "offset": None}),
    "drift": ("Slow drift", "Linear ramp on the target channel(s): the classic valid-but-wrong walk-off.",
              ["falcon9", "quad", "spacecraft"], {"channels": [], "rate": None}),
    "replay": ("Replay", "Replays a recorded window of earlier true frames under the current seq/t.",
               ["falcon9", "quad", "spacecraft"], {"window_s": 6.0}),
    "freeze": ("Freeze", "Holds the last values before the attack: a stuck sensor / frozen link.",
               ["falcon9", "quad", "spacecraft"], {}),
    "coordinated": ("Coordinated", "Kinematically self-consistent fake trajectory that beats naive "
                    "per-channel physics rules.", ["falcon9", "quad"], {"channels": []}),
    "ppo": ("PPO (learned)", "RL policy trained to maximise operator-picture error while staying under "
            "the detector's alert threshold.", ["falcon9", "quad"], {}),
    "ppo_r2": ("PPO round 2", "Arms race round 2: a PPO policy retrained against the hardened detector "
               "(round-1 discriminator).", ["falcon9"], {}),
}

# Learned attacks: name -> model file suffix (models/ppo_<schema><suffix>.zip).
_LEARNED = {"ppo": "", "ppo_r2": "_r2"}


def _model_path(schema_name: str, attack: str = "ppo") -> Path:
    return MODELS / f"ppo_{schema_name}{_LEARNED[attack]}.zip"


def list_attacks() -> list[dict]:
    out = []
    for name, (label, desc, schemas, extra) in _META.items():
        entry: dict[str, Any] = {
            "name": name, "label": label, "description": desc, "schemas": schemas,
            "params": {**_COMMON, **extra},
        }
        if name in _LEARNED:
            entry["available"] = any(_model_path(s, name).exists() for s in schemas)
        out.append(entry)
    return out


def make_attack(name: str, schema: ChannelSchema, **params: Any) -> _Attack:
    """Build an attack. Unknown params are dropped so the console can pass a superset safely."""
    if name == "ppo":
        from vbw.attacker.env import PpoAttack  # lazy: avoids importing SB3 unless needed

        return PpoAttack(schema=schema, **_filter(PpoAttack, params))
    if name == "ppo_r2":
        from vbw.attacker.env import PpoR2Attack

        return PpoR2Attack(schema=schema, **_filter(PpoR2Attack, params))
    cls = SCRIPTED.get(name)
    if cls is None:
        raise KeyError(f"unknown attack {name!r}; have {sorted(_META)}")
    return cls(schema=schema, **_filter(cls, params))


def _filter(cls: type, params: dict[str, Any]) -> dict[str, Any]:
    fields = getattr(cls, "__dataclass_fields__", {})
    return {k: v for k, v in params.items() if k in fields and k != "schema"}
