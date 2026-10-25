"""Export one schema's detector configuration as a flat text file for the C++ edge monitor (edge/SPEC.md section 2).

    uv run python scripts/export_edge_config.py --schema falcon9      # -> edge/config/falcon9.cfg
    uv run python scripts/export_edge_config.py --schema all

The C++ build has no JSON library, so everything vbw_edge needs to reproduce the Python BaseDetector bit for bit is
written as `key value value ...` lines: rule thresholds (defaults merged exactly as L1Config.load does), rule
lookbacks, every leaf of the calibrated L1 params (nested dicts flattened to dotted keys), the module constants the
rules use, the L2 normalization arrays, and the alert config. Floats are written with repr(), which round-trips
exactly through strtod, so the C++ sees the same doubles as Python. Paths are relative to the cfg file's directory.

The output has no timestamp, so re-exporting unchanged models gives a byte-identical file (tests rely on that to
catch a stale cfg after recalibration).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path

import vbw.detector.physics as physics
from vbw.detector.alerting import AlertConfig
from vbw.detector.l2 import MODELS_DIR, Z_CLIP, L2Norm, model_paths
from vbw.detector.physics import NOMINAL_RATE_HZ, RULES_BY_SCHEMA, L1Config
from vbw.schema import SCHEMAS_BY_NAME, ChannelSchema

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "edge" / "config"
EDGE_SCHEMAS = ("falcon9", "quad")
CONST_PREFIXES = ("F9_", "QUAD_")
CONST_EXTRA = ("G0", "R_EARTH")
# L2Model floors the clean residual MAD at this before dividing (inline literal in vbw/detector/l2.py).
L2_MAD_FLOOR = 1e-6

Value = int | float | str


def fmt(value: Value) -> str:
    """One config token. repr() of a float round-trips exactly; bools become 0/1 so the line stays numeric."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)  # 'nan', 'inf', '-inf' are all strtod-parseable
    text = str(value)
    if not text or any(ch.isspace() for ch in text) or "#" in text:
        raise ValueError(f"string value {text!r} can't be written as one config token")
    return text


def flatten(tree: dict, prefix: str = "") -> dict[str, list[Value]]:
    """Nested dict -> {dotted.key: values}: dicts recurse, lists become value lists, scalars one value."""
    out: dict[str, list[Value]] = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, name))
        elif isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"empty list at {name!r}: a config line needs at least one value")
            if any(isinstance(v, (dict, list, tuple)) for v in value):
                raise ValueError(f"nested containers inside the list at {name!r} can't be flattened")
            out[name] = list(value)
        else:
            out[name] = [value]
    return out


def rule_constants() -> dict[str, Value]:
    """Every module-level numeric F9_* / QUAD_* constant in vbw.detector.physics, plus G0 and R_EARTH."""
    names = sorted(n for n in vars(physics) if n.startswith(CONST_PREFIXES)) + list(CONST_EXTRA)
    out: dict[str, Value] = {}
    for name in names:
        value = getattr(physics, name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = value
    return out


def config_entries(schema: ChannelSchema, cfg_dir: Path, models_dir: Path = MODELS_DIR) -> list[tuple[str, list[Value]]]:
    """Ordered (key, values) entries of the cfg for `schema`, as the Python detector would load them."""
    rules = RULES_BY_SCHEMA[schema.name]
    l1 = L1Config.load(schema)
    entries: list[tuple[str, list[Value]]] = [
        ("schema", [schema.name]),
        ("schema_id", [schema.schema_id]),
        ("rate_hz", [NOMINAL_RATE_HZ[schema.name]]),
        ("channels", list(schema.channels)),
        ("l1.rules", [r.name for r in rules]),
        ("l1.calibrated", [l1.calibrated]),
    ]
    entries += [(f"l1.threshold.{r.name}", [float(l1.thresholds[r.name])]) for r in rules]
    entries += [(f"l1.lookback.{r.name}", [r.lookback]) for r in rules]
    entries += [(f"l1.params.{k}", v) for k, v in flatten(l1.params).items()]
    entries += [(f"l1.const.{k}", [v]) for k, v in rule_constants().items()]

    onnx_path, norm_path = model_paths(schema, models_dir)
    if onnx_path.exists() and norm_path.exists():
        norm = L2Norm.load(norm_path)
        if list(norm.channels) != list(schema.channels):
            raise ValueError(f"{norm_path.name} channels {norm.channels} != schema {schema.channels}")
        rel = Path(os.path.relpath(onnx_path, cfg_dir)).as_posix()
        entries += [
            ("l2.onnx", [rel]),
            ("l2.window", [norm.window]),
            ("l2.mean", list(norm.mean)),
            ("l2.std", list(norm.std)),
            ("l2.dstd", list(norm.dstd)),
            ("l2.res_median", list(norm.res_median)),
            ("l2.res_mad", list(norm.res_mad)),
            ("l2.mad_floor", [L2_MAD_FLOOR]),
            ("l2.inst_threshold", [float(norm.inst_threshold)]),
            ("l2.ewma_alpha", [float(norm.ewma_alpha)]),
            ("l2.ewma_threshold", [float(norm.ewma_threshold)]),
            ("l2.z_clip", [float(Z_CLIP)]),
        ]

    alert = AlertConfig.load(schema.name)
    entries += [
        ("alert.tau", [float(alert.tau)]),
        ("alert.k", [int(alert.k)]),
        ("alert.n", [int(alert.n)]),
        ("alert.clear_frames", [int(alert.clear_frames)]),
        ("alert.clear_ratio", [float(alert.clear_ratio)]),
        ("alert.warmup", [int(alert.warmup)]),
    ]
    return entries


def render(schema: ChannelSchema, entries: Iterable[tuple[str, list[Value]]]) -> str:
    lines = [
        f"# vbw_edge config for schema {schema.name}: generated by scripts/export_edge_config.py, do not edit.",
        "# Format (edge/SPEC.md section 2): `key value value ...`; floats are Python repr() and round-trip exactly.",
        "# Sources: models/l1_thresholds.json (defaults merged as L1Config.load), vbw.detector.physics constants,",
        f"# models/l2_{schema.name}_norm.json, models/l1_alert.json. Paths are relative to this file.",
    ]
    section = None
    for key, values in entries:
        head = key.split(".", 1)[0] if "." in key else ""
        if head != section:
            lines.append("")
            section = head
        lines.append(" ".join([key, *map(fmt, values)]))
    return "\n".join(lines) + "\n"


def export(schema_name: str, out: Path | None = None) -> Path:
    schema = SCHEMAS_BY_NAME[schema_name]
    out = out or CONFIG_DIR / f"{schema_name}.cfg"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render(schema, config_entries(schema, out.parent.resolve()))
    out.write_text(text, encoding="utf-8", newline="\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", default="all", choices=[*EDGE_SCHEMAS, "all"])
    ap.add_argument("--out", type=Path, default=None, help="output path (single schema only)")
    args = ap.parse_args()
    names = EDGE_SCHEMAS if args.schema == "all" else (args.schema,)
    if args.out is not None and len(names) > 1:
        ap.error("--out needs a single --schema")
    for name in names:
        path = export(name, args.out)
        n = sum(1 for line in path.read_text().splitlines() if line and not line.startswith("#"))
        print(f"{name}: {n} entries -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
