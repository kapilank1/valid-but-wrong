"""Score combination and alert persistence / hysteresis, shared by every detector.

An alert rises when the score exceeds τ on at least k of the last n frames, and clears only after m consecutive
frames below τ_clear (< τ). A single glitchy frame never alerts; a real attack doesn't flicker on and off.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALERT_PATH = ROOT / "models" / "l1_alert.json"


def soft_ratio(ratio: float) -> float:
    """Map a ratio-to-threshold into 0..1, with exactly 0.5 at the threshold (r²/(1+r²))."""
    r2 = ratio * ratio
    return r2 / (1.0 + r2)


def combine(*scores: float) -> float:
    """Noisy-OR: two half-suspicious layers are more suspicious than either alone."""
    keep = 1.0
    for s in scores:
        keep *= 1.0 - min(max(s, 0.0), 1.0)
    return 1.0 - keep


@dataclass
class AlertConfig:
    tau: float = 0.5                 # score needed to count a frame as suspicious
    k: int = 3                       # suspicious frames needed ...
    n: int = 5                       # ... within the last n frames to raise
    clear_frames: int = 20           # consecutive quiet frames needed to clear
    clear_ratio: float = 0.8         # quiet means score < clear_ratio · tau
    warmup: int = 10                 # frames after reset that can never alert
    meta: dict | None = None

    @classmethod
    def load(cls, schema_name: str, path: Path = ALERT_PATH) -> "AlertConfig":
        if path.exists():
            blob = json.loads(path.read_text()).get(schema_name)
            if blob:
                return cls(**blob)
        return cls()

    def save(self, schema_name: str, path: Path = ALERT_PATH) -> None:
        blob = json.loads(path.read_text()) if path.exists() else {}
        blob[schema_name] = asdict(self)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(blob, indent=1))


class AlertState:
    """k-of-n raise, m-quiet-frames clear. Pure state machine over a stream of scores."""

    def __init__(self, config: AlertConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._recent: deque[bool] = deque(maxlen=self.config.n)
        self._quiet = 0
        self._frames = 0
        self.active = False

    def update(self, score: float) -> bool:
        cfg = self.config
        self._frames += 1
        if self._frames <= cfg.warmup:
            return False
        self._recent.append(score > cfg.tau)
        if not self.active:
            if sum(self._recent) >= cfg.k:
                self.active, self._quiet = True, 0
        else:
            self._quiet = self._quiet + 1 if score < cfg.clear_ratio * cfg.tau else 0
            if self._quiet >= cfg.clear_frames:
                self.active = False
                self._recent.clear()
        return self.active


def alert_series(scores, config: AlertConfig) -> list[bool]:
    """Offline replay of AlertState over a score sequence (used by calibration and tests)."""
    state = AlertState(config)
    return [state.update(float(s)) for s in scores]
