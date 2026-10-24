"""Asyncio runner: plays one scenario through the link + MITM + detector at real time x speed.

Control semantics (POST /api/control):
- start: plays. Loads `scenario` first if it differs from the loaded one, nothing is loaded, or the last run
  finished (each load is a new run with fresh counters and a `scenario` message). Otherwise resumes where
  it stopped. Other fields (attack, attack_params, detector, speed) are applied, hot-swapping if mid-run.
- stop: pauses at the current frame. The run stays loaded; `start` resumes it.
- reset: stops and reloads the scenario from its first frame (a new run), without playing.

Hot swaps happen at a frame boundary. A swapped-in detector is warmed with the last `warm_frames` reported
frames (and their L0); a swapped-in attack is warmed with the last true frames and, unless the caller gave
`start_t`, starts at the current mission time. Exceptions in any stage are logged and counted, never fatal.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from vbw.pipeline import messages
from vbw.pipeline.callouts import spoken_line
from vbw.pipeline.registries import Registries, safe_list
from vbw.pipeline.session import DEFAULT_WARM_FRAMES, Session, Tick
from vbw.schema import Attack, ChannelSchema, Frame, Source
from vbw.store.base import NullStore, RunRecord, Store, TickRecord

log = logging.getLogger("vbw.pipeline")

Publisher = Callable[[dict[str, Any]], None]
ACTIONS = ("start", "stop", "reset")


class PipelineError(ValueError):
    """A bad request (unknown scenario/attack/detector, bad action). The server maps it to HTTP 400."""


@dataclass
class Status:
    running: bool = False
    scenario: str | None = None
    attack: str = "none"
    attack_params: dict[str, Any] = field(default_factory=dict)
    detector: str = "base"
    speed: float = 1.0
    t: float = 0.0
    frames: int = 0
    alerts: int = 0
    # Additive fields beyond docs/CONTRACT.md:
    run_id: str | None = None
    finished: bool = False            # source exhausted; the next start loads a new run
    attacked_frames: int = 0          # frames where reported != truth
    l0_failures: int = 0              # frames that failed a link-layer check
    errors: int = 0                   # exceptions caught and survived

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunnerConfig:
    speed_min: float = 0.5
    speed_max: float = 20.0
    warm_frames: int = DEFAULT_WARM_FRAMES
    status_interval_s: float = 1.0    # periodic status broadcast while running
    max_lag_s: float = 0.5            # further behind schedule than this -> re-anchor instead of bursting
    max_sleep_s: float = 2.0          # data gaps longer than this (wall time) are shortened
    stop_timeout_s: float = 2.0


def pace_delay(t: float, anchor_t: float, anchor_wall: float, speed: float, now: float) -> float:
    """Seconds until mission time `t` is due, given that `anchor_t` played at wall time `anchor_wall`."""
    return anchor_wall + (t - anchor_t) / speed - now


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class _ErrorLog:
    """Logs the first few errors with tracebacks, then every 100th, so a hot loop can't flood the log."""

    def __init__(self, first: int = 5, every: int = 100):
        self.first, self.every, self.count = first, every, 0

    def __call__(self, message: str, exc: BaseException | None = None) -> None:
        self.count += 1
        if self.count <= self.first or self.count % self.every == 0:
            log.error("pipeline error #%d: %s", self.count, message, exc_info=exc)


class Runner:
    def __init__(
        self,
        registries: Registries,
        publish: Publisher,
        store: Store | None = None,
        config: RunnerConfig = RunnerConfig(),
        seed: int | None = None,
    ):
        self.registries = registries
        self.publish = publish
        self.store: Store = store or NullStore()
        self.config = config
        self.seed = seed
        self.status = Status()
        self.scenario_info: dict[str, Any] | None = None
        self._session: Session | None = None
        self._pending: Frame | None = None  # a frame pulled from the source but not yet played (stopped mid-wait)
        self._run: RunRecord | None = None
        self._run_recorded = False
        self._attack_request: tuple[str, dict[str, Any]] = ("none", {})
        self._alerting = False
        self._task: asyncio.Task | None = None
        self._running = False
        self._wake = asyncio.Event()
        self._control = asyncio.Lock()
        self._frame_lock = asyncio.Lock()
        self._anchor: tuple[float, float] | None = None
        self._last_status_wall = 0.0
        self._catalog: dict[str, list[dict]] = {}
        self._errors = _ErrorLog()

    # ================================================================== public API

    async def control(
        self,
        action: str,
        scenario: str | None = None,
        attack: str | None = None,
        attack_params: dict[str, Any] | None = None,
        detector: str | None = None,
        speed: float | None = None,
        seed: int | None = None,
    ) -> Status:
        if action not in ACTIONS:
            raise PipelineError(f"unknown action {action!r}; expected one of {ACTIONS}")
        async with self._control:
            if seed is not None:
                self.seed = seed
            if speed is not None:
                self._set_speed(speed)
            if action == "stop":
                await self._stop_loop()
                await self._apply_swaps(attack, attack_params, detector)
            elif action == "reset":
                await self._stop_loop()
                await self._load(scenario or self.status.scenario or await self._default_scenario(),
                                 attack, attack_params, detector)
            else:
                await self._start(scenario, attack, attack_params, detector)
        self._publish_status()
        return self.status

    async def set_attack(self, name: str, params: dict[str, Any] | None = None) -> Status:
        async with self._control:
            await self._swap_attack(name, params)
        self._publish_status()
        return self.status

    async def set_detector(self, name: str) -> Status:
        async with self._control:
            await self._swap_detector(name)
        self._publish_status()
        return self.status

    async def catalog(self) -> dict[str, list[dict]]:
        """Catalog from the registries, cached once each list is non-empty."""
        for key, fn in (("scenarios", self.registries.list_scenarios),
                        ("attacks", self.registries.list_attacks),
                        ("detectors", self.registries.list_detectors)):
            if not self._catalog.get(key):
                self._catalog[key] = await asyncio.to_thread(safe_list, fn, key)
        return dict(self._catalog)

    async def wait_stopped(self, timeout: float | None = None) -> None:
        """Wait for the play loop to exit (end of scenario or stop). Mainly for tests and scripts."""
        task = self._task
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout)

    async def shutdown(self) -> None:
        async with self._control:
            await self._stop_loop()
            self._end_run()

    # ================================================================== control internals

    async def _start(self, scenario: str | None, attack: str | None,
                     attack_params: dict[str, Any] | None, detector: str | None) -> None:
        target = scenario or self.status.scenario or await self._default_scenario()
        if self._session is None or target != self.status.scenario or self.status.finished:
            await self._stop_loop()
            await self._load(target, attack, attack_params, detector)
        else:
            await self._apply_swaps(attack, attack_params, detector)
        self._launch()

    async def _apply_swaps(self, attack: str | None, attack_params: dict[str, Any] | None,
                           detector: str | None) -> None:
        if attack is not None or attack_params is not None:
            await self._swap_attack(attack or self.status.attack, attack_params)
        if detector is not None and detector != self.status.detector:
            await self._swap_detector(detector)

    async def _load(self, scenario: str, attack: str | None, attack_params: dict[str, Any] | None,
                    detector: str | None) -> None:
        """Build a fresh session for `scenario` (a new run). Leaves the runner stopped."""
        if attack is not None or attack_params is not None:
            request = (attack or self._attack_request[0], dict(attack_params or {}))
        else:
            request = self._attack_request
        attack_name, params = request
        detector_name = detector or self.status.detector
        catalog = await self.catalog()
        try:
            session = await asyncio.to_thread(self._build_session, scenario, attack_name, params, detector_name)
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface registry errors as a 400 with the reason
            raise PipelineError(f"could not load {scenario!r}: {type(exc).__name__}: {exc}") from exc

        self._end_run()
        self._session = session
        self._attack_request = (attack_name, dict(params))
        self.scenario_info = scenario_info(scenario, session.source, catalog.get("scenarios", []))
        self.status = Status(
            running=False, scenario=scenario, attack=attack_name,
            attack_params=effective_params(session.attack, params, self._param_names(attack_name)),
            detector=detector_name,
            speed=self.status.speed, run_id=uuid.uuid4().hex,
        )
        self._run = None
        self._run_recorded = False
        self._pending = None
        self._alerting = False
        self._anchor = None
        self.publish(messages.scenario(self.scenario_info))

    def _build_session(self, scenario: str, attack_name: str, params: dict[str, Any],
                       detector_name: str) -> Session:
        source: Source = self.registries.make_source(scenario, seed=self.seed)
        attack = self._make_attack(attack_name, source.schema, params)
        detector = self.registries.make_detector(detector_name, source.schema)
        return Session(source, attack, detector, warm_frames=self.config.warm_frames)

    def _make_attack(self, name: str, schema: ChannelSchema, params: dict[str, Any]) -> Attack:
        entry = _find(self._catalog.get("attacks", []), "name", name)
        supported = entry.get("schemas") if entry else None
        if supported and schema.name not in supported:
            raise PipelineError(f"attack {name!r} does not support schema {schema.name!r} (supports {supported})")
        attack = self.registries.make_attack(name, schema, **params)
        attack.reset(self.seed)
        return attack

    def _attack_kwargs(self, name: str, params: dict[str, Any] | None) -> dict[str, Any]:
        kwargs = dict(params or {})
        entry = _find(self._catalog.get("attacks", []), "name", name)
        accepts_start = bool(entry) and "start_t" in (entry.get("params") or {})
        if self.status.frames > 0 and "start_t" not in kwargs and accepts_start:
            kwargs["start_t"] = self.status.t  # mid-run swap: attack now, not at the scenario default
        return kwargs

    async def _swap_attack(self, name: str, params: dict[str, Any] | None) -> None:
        await self.catalog()
        self._check_known("attacks", "name", name)
        if self._session is None:
            self.status.attack, self.status.attack_params = name, dict(params or {})
            self._attack_request = (name, dict(params or {}))
            return
        kwargs = self._attack_kwargs(name, params)
        schema = self._session.schema
        try:
            attack = await asyncio.to_thread(self._make_attack, name, schema, kwargs)
        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"could not make attack {name!r}: {type(exc).__name__}: {exc}") from exc
        async with self._frame_lock:
            await asyncio.to_thread(self._session.swap_attack, attack)
            self.status.attack = name
            self.status.attack_params = effective_params(attack, kwargs, self._param_names(name))
            self._attack_request = (name, dict(params or {}))  # what a reset replays (no injected start_t)
        log.info("attack -> %s %s", name, self.status.attack_params)

    async def _swap_detector(self, name: str) -> None:
        await self.catalog()
        self._check_known("detectors", "name", name)
        if self._session is None:
            self.status.detector = name
            return
        schema = self._session.schema
        try:
            detector = await asyncio.to_thread(self.registries.make_detector, name, schema)
        except Exception as exc:  # noqa: BLE001
            raise PipelineError(f"could not make detector {name!r}: {type(exc).__name__}: {exc}") from exc
        async with self._frame_lock:
            warmed = await asyncio.to_thread(self._session.swap_detector, detector)
            self.status.detector = name
        log.info("detector -> %s (warmed on %d frames)", name, warmed)

    def _param_names(self, attack_name: str) -> list[str]:
        entry = _find(self._catalog.get("attacks", []), "name", attack_name) or {}
        return list(entry.get("params") or {})

    def _check_known(self, kind: str, key: str, name: str) -> None:
        entries = self._catalog.get(kind) or []
        if entries and _find(entries, key, name) is None:
            known = [e.get(key) for e in entries]
            raise PipelineError(f"unknown {kind[:-1]} {name!r}; known: {known}")

    async def _default_scenario(self) -> str:
        scenarios = (await self.catalog()).get("scenarios", [])
        if not scenarios:
            raise PipelineError("no scenario given and the scenario catalog is empty")
        hero = next((s for s in scenarios if s.get("hero")), None)
        falcon = next((s for s in scenarios if str(s.get("id", "")).startswith("falcon9:")), None)
        return str((hero or falcon or scenarios[0])["id"])

    def _set_speed(self, speed: float) -> None:
        self.status.speed = clamp(float(speed), self.config.speed_min, self.config.speed_max)
        if self._anchor is not None:
            self._anchor = (self.status.t, asyncio.get_running_loop().time())
        self._wake.set()

    # ================================================================== play loop

    def _launch(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self.status.running = True
        self._anchor = None
        self._task = asyncio.create_task(self._play(), name="vbw-runner")

    async def _stop_loop(self) -> None:
        task = self._task
        self._running = False
        self.status.running = False
        self._wake.set()
        if task is None or task.done():
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), self.config.stop_timeout_s)
        except asyncio.TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _is_live(self) -> bool:
        info = self.scenario_info or {}
        return bool(info.get("live")) or str(self.status.scenario or "").startswith("sitl:")

    async def _play(self) -> None:
        session = self._session
        assert session is not None
        live = self._is_live()
        loop = asyncio.get_running_loop()
        self._last_status_wall = loop.time()
        try:
            while self._running:
                truth, self._pending = self._pending, None
                try:
                    if truth is None:
                        truth = await asyncio.to_thread(session.next_truth) if live else session.next_truth()
                except Exception as exc:  # noqa: BLE001 - a dead source ends the run
                    self._count_error(f"source: {type(exc).__name__}: {exc}", exc)
                    self._finish()
                    break
                if truth is None:
                    self._finish()
                    break
                if not live and not await self._wait_until_due(truth.t):
                    self._pending = truth  # stopped while waiting: play this frame first on resume
                    break
                try:
                    async with self._frame_lock:
                        self._handle(session.step(truth))
                except Exception as exc:  # noqa: BLE001 - log and keep going
                    self._count_error(f"frame seq={truth.seq}: {type(exc).__name__}: {exc}", exc)
                if live:
                    await asyncio.sleep(0)
                if loop.time() - self._last_status_wall >= self.config.status_interval_s:
                    self._last_status_wall = loop.time()
                    self._publish_status()
        finally:
            self._running = False
            self.status.running = False

    async def _wait_until_due(self, t: float) -> bool:
        """Sleep until frame time `t` is due at the current speed. False if stopped meanwhile."""
        loop = asyncio.get_running_loop()
        while self._running:
            now = loop.time()
            speed = self.status.speed
            if self._anchor is None:
                self._anchor = (t, now)
            delay = pace_delay(t, *self._anchor, speed, now)
            if delay < -self.config.max_lag_s:
                self._anchor = (t, now)  # fell behind (slow step, swap): resync instead of bursting
                delay = 0.0
            if delay > self.config.max_sleep_s:
                self._anchor = (t - self.config.max_sleep_s * speed, now)  # shorten long data gaps
                delay = self.config.max_sleep_s
            if delay <= 0.0005:
                await asyncio.sleep(0)  # always yield so WebSocket senders and HTTP stay responsive
                return self._running
            if delay < 0.05:
                await asyncio.sleep(delay)
                return self._running
            self._wake.clear()
            try:
                async with asyncio.timeout(delay):
                    await self._wake.wait()
            except TimeoutError:
                return self._running
            # woken by a speed change or stop: loop and recompute
        return False

    def _handle(self, tick: Tick) -> None:
        status = self.status
        status.frames += 1
        status.t = tick.truth.t
        status.attacked_frames += int(tick.attack_active)
        status.l0_failures += int(not tick.l0.ok)
        for error in tick.errors:
            self._count_error(error)
        verdict = tick.verdict
        if verdict is None:
            return
        self.publish(messages.tick(tick.truth, tick.reported, verdict, status.attack, tick.attack_active))
        self._record(tick)
        if verdict.alert and not self._alerting:
            status.alerts += 1
            schema_name = self._session.schema.name if self._session else ""
            self.publish(messages.alert(verdict, spoken_line(verdict, schema_name)))
        self._alerting = bool(verdict.alert)

    def _finish(self) -> None:
        self._running = False
        self.status.running = False
        self.status.finished = True
        self._end_run()
        self._publish_status()

    # ================================================================== store + status

    def _record(self, tick: Tick) -> None:
        if self._run is None:
            session = self._session
            assert session is not None
            self._run = RunRecord(
                run_id=self.status.run_id or uuid.uuid4().hex,
                started_at=datetime.now(timezone.utc),
                scenario=self.status.scenario or "",
                vehicle_id=session.vehicle_id,
                schema=session.schema.name,
                attack=self.status.attack,
                attack_params=dict(self.status.attack_params),
                detector=self.status.detector,
                speed=self.status.speed,
                seed=self.seed,
            )
        try:
            if not self._run_recorded:
                self.store.record_run_start(self._run)
                self._run_recorded = True
            assert tick.verdict is not None
            self.store.record_tick(TickRecord(self._run, tick.truth, tick.reported, tick.verdict,
                                              self.status.attack, tick.attack_active))
        except Exception as exc:  # noqa: BLE001 - the store must never break the loop
            self._count_error(f"store: {type(exc).__name__}: {exc}", exc)

    def _end_run(self) -> None:
        if self._run is not None and self._run_recorded:
            try:
                self.store.record_run_end(self._run.run_id, datetime.now(timezone.utc),
                                          self.status.frames, self.status.alerts)
            except Exception as exc:  # noqa: BLE001
                self._count_error(f"store: {type(exc).__name__}: {exc}", exc)
            self._run_recorded = False

    def _count_error(self, message: str, exc: BaseException | None = None) -> None:
        self.status.errors += 1
        self._errors(message, exc)

    def _publish_status(self) -> None:
        self.publish(messages.status(self.status.to_dict()))


# ====================================================================== helpers


def _find(entries: list[dict], key: str, value: str) -> dict | None:
    return next((e for e in entries if e.get(key) == value), None)


def effective_params(attack: Attack, given: dict[str, Any], names: Iterable[str] = ()) -> dict[str, Any]:
    """Params in effect: the attack's `params` dict if it has one, else its attributes named in the catalog
    entry (defaults resolved by the attack itself), overlaid with what the caller gave."""
    params = getattr(attack, "params", None)
    if isinstance(params, dict):
        return {**params, **given}
    resolved = {name: getattr(attack, name) for name in names if hasattr(attack, name)}
    return {**resolved, **given}


def scenario_info(scenario_id: str, source: Source, catalog: list[dict]) -> dict[str, Any]:
    """ScenarioInfo for the `scenario` message: the catalog entry, with source facts as fallbacks."""
    entry = _find(catalog, "id", scenario_id) or {}
    fallback = {
        "id": scenario_id,
        "label": scenario_id,
        "schema": source.schema.name,
        "vehicle_id": source.vehicle_id,
        "rate_hz": source.rate_hz,
        "duration_s": None,
        "events": {},
        "description": "",
    }
    return {**fallback, **entry, "id": scenario_id}
