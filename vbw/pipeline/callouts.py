"""Mission-control voice callouts built from a Verdict.

One short line per alert rising edge, e.g.
"Flight, Guidance. Altitude disagrees with vertical velocity. Possible spoofed telemetry. Confidence 94 percent."
The console sends it to TTS, so it avoids symbols and units that read badly aloud.
"""

from __future__ import annotations

import math

from vbw.schema import Verdict

# Console position that would make the call, by schema name.
STATIONS: dict[str, str] = {"falcon9": "Guidance", "quad": "GNC", "spacecraft": "INCO"}

SPOKEN_CHANNELS: dict[str, dict[str, str]] = {
    "falcon9": {
        "speed": "speed", "altitude": "altitude", "vz": "vertical velocity", "vx": "horizontal velocity",
        "downrange": "downrange distance", "accel": "acceleration", "angle": "flight path angle",
        "q": "dynamic pressure",
    },
    "quad": {
        "pn": "north position", "pe": "east position", "alt": "altitude",
        "vn": "north velocity", "ve": "east velocity", "vu": "climb rate",
        "roll": "roll", "pitch": "pitch", "yaw": "heading",
        "p": "roll rate", "q": "pitch rate", "r": "yaw rate",
        "ax": "forward acceleration", "ay": "lateral acceleration", "az": "vertical acceleration",
        "baro_alt": "barometric altitude", "groundspeed": "groundspeed", "battery_v": "battery voltage",
    },
    "spacecraft": {"value": "telemetry channel"},
}


def spoken_channel(channel: str, schema_name: str) -> str:
    return SPOKEN_CHANNELS.get(schema_name, {}).get(channel, channel.replace("_", " "))


def _severity(value: float, threshold: float) -> float:
    if not threshold or not math.isfinite(value) or not math.isfinite(threshold):
        return abs(value) if math.isfinite(value) else 0.0
    return abs(value) / abs(threshold)


def _sentence(text: str) -> str:
    text = text.strip()
    return text[0].upper() + text[1:] + ("" if text.endswith(".") else ".") if text else text


def _l1_sentence(verdict: Verdict, schema_name: str) -> str:
    worst = max(verdict.l1_violations, key=lambda v: _severity(v.value, v.threshold))
    names = [spoken_channel(c, schema_name) for c in worst.channels]
    if len(names) >= 2:
        return _sentence(f"{names[0]} disagrees with {names[1]}")
    return _sentence(f"{names[0]} outside physical limits") if names else ""


def finding(verdict: Verdict, schema_name: str) -> str:
    """The single most important thing wrong with this frame, as one spoken sentence.

    Same priority as the detector's written explanation (BaseDetector._explain): the physics violation unless the
    learned layer scored higher, so the voice and the event log never blame different things.
    """
    finite = {c: abs(z) for c, z in verdict.l2_residuals.items() if z is not None and math.isfinite(z)}
    l2_leads = bool(finite) and verdict.l2_score > 0 and verdict.l2_score > verdict.l1_score
    if verdict.l1_violations and not l2_leads and (line := _l1_sentence(verdict, schema_name)):
        return line
    if finite and verdict.l2_score >= max(verdict.l1_score, verdict.l3_prob or 0.0):
        channel = max(finite, key=finite.get)
        return _sentence(f"{spoken_channel(channel, schema_name)} departing from predicted track")
    if verdict.l1_violations and (line := _l1_sentence(verdict, schema_name)):
        return line
    if verdict.l3_prob is not None and verdict.l3_prob >= 0.5:
        return "Frame pattern matches known forgeries."
    if not verdict.l0.ok:
        return "Link integrity failure."
    return "Telemetry inconsistent with vehicle dynamics."


def spoken_line(verdict: Verdict, schema_name: str) -> str:
    station = STATIONS.get(schema_name, "Telemetry")
    parts = [f"Flight, {station}.", finding(verdict, schema_name)]
    if verdict.l0.ok:
        parts.append("Possible spoofed telemetry.")
    if math.isfinite(verdict.score) and verdict.score > 0:
        parts.append(f"Confidence {round(100 * min(verdict.score, 1.0))} percent.")
    return " ".join(parts)
