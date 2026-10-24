"""FastAPI ground-station server (docs/CONTRACT.md). Run: `uv run uvicorn vbw.pipeline.server:app --port 8000`.

Contract endpoints: /api/health, /api/catalog, /api/status, /api/control, /api/attack, /api/detector,
/api/bench and the /ws feed. Tiger Data endpoints (return {"available": false} without DATABASE_URL):
- GET /api/runs?limit=50          recent runs with peak score, alert/attacked/false-alert frame counts
- GET /api/runs/{run_id}/timeline per-second peak score + counts from the verdicts_1s continuous aggregate,
                                  plus attack onset, first alert and detection latency for the run
- GET /api/analytics              time-to-first-alert per (attack, detector) across runs + detector scorecard
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Literal

import anyio
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from vbw.pipeline import messages
from vbw.pipeline.hub import Hub
from vbw.pipeline.jsonsafe import dumps
from vbw.pipeline.registries import Registries, default_registries
from vbw.pipeline.runner import PipelineError, Runner, RunnerConfig
from vbw.store.base import Store

log = logging.getLogger("vbw.pipeline")

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCH_PATH = ROOT / "results" / "bench_summary.json"
DEFAULT_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


class SafeJSONResponse(JSONResponse):
    """JSON response that turns NaN/inf into null and handles numpy/datetime values."""

    def render(self, content: Any) -> bytes:
        return dumps(content).encode("utf-8")


class ControlBody(BaseModel):
    action: Literal["start", "stop", "reset"]
    scenario: str | None = None
    attack: str | None = None
    attack_params: dict[str, Any] | None = None
    detector: str | None = None
    speed: float | None = None
    seed: int | None = None


class AttackBody(BaseModel):
    attack: str
    attack_params: dict[str, Any] | None = None


class DetectorBody(BaseModel):
    detector: str


def cors_origins(env: dict[str, str] | None = None) -> list[str]:
    """localhost:3000 plus any comma-separated origins in VBW_CORS_ORIGINS (e.g. the deployed console)."""
    env = dict(os.environ) if env is None else env
    extra = [o.strip() for o in env.get("VBW_CORS_ORIGINS", "").split(",") if o.strip()]
    return list(dict.fromkeys([*DEFAULT_ORIGINS, *extra]))


def read_bench(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"available": False}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return {"available": False, "error": f"unreadable {path.name}: {exc}"}
    return data if isinstance(data, dict) else {"available": True, "results": data}


async def _preload_policies() -> None:
    """Load the PPO attacker policies in the background at startup, so the first PPO attack starts on time."""
    try:
        from vbw.attacker.env import preload_policies

        loaded = await asyncio.to_thread(preload_policies)
        log.info("attacker policies preloaded: %s", ", ".join(loaded) or "none")
    except Exception as exc:  # noqa: BLE001 - optional warm-up; attacks still load on demand
        log.warning("attacker policy preload failed: %s", exc)


def create_app(
    registries: Registries | None = None,
    store: Store | None = None,
    bench_path: Path | None = None,
    runner_config: RunnerConfig = RunnerConfig(),
    hub: Hub | None = None,
) -> FastAPI:
    """Build the app. Defaults: real registries (imported lazily), store from DATABASE_URL, results/ bench."""
    bench_file = bench_path or Path(os.environ.get("VBW_BENCH_PATH", DEFAULT_BENCH_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if store is None:
            from vbw.store.timescale import store_from_env

            app.state.store = store_from_env()
        else:
            app.state.store = store
        app.state.hub = hub or Hub()
        app.state.runner = Runner(registries or default_registries(), app.state.hub.publish,
                                  app.state.store, runner_config)
        await app.state.store.start()
        log.info("pipeline up (store: %s)", type(app.state.store).__name__)
        preload = asyncio.create_task(_preload_policies()) if registries is None else None
        try:
            yield
        finally:
            if preload is not None and not preload.done():
                preload.cancel()
            await app.state.runner.shutdown()
            await app.state.store.close()

    app = FastAPI(title="Valid-But-Wrong pipeline", default_response_class=SafeJSONResponse, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins(), allow_methods=["*"], allow_headers=["*"])

    def runner() -> Runner:
        return app.state.runner

    async def guarded(coro: Any) -> SafeJSONResponse:
        try:
            status = await coro
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SafeJSONResponse(status.to_dict())

    # ------------------------------------------------------------------ contract endpoints

    @app.get("/api/health")
    async def health() -> SafeJSONResponse:
        return SafeJSONResponse({"ok": True, "db": app.state.store.stats(), "ws": app.state.hub.stats()})

    @app.get("/api/catalog")
    async def catalog() -> SafeJSONResponse:
        return SafeJSONResponse(await runner().catalog())

    @app.get("/api/status")
    async def status() -> SafeJSONResponse:
        return SafeJSONResponse(runner().status.to_dict())

    @app.post("/api/control")
    async def control(body: ControlBody) -> SafeJSONResponse:
        return await guarded(runner().control(**body.model_dump()))

    @app.post("/api/attack")
    async def attack(body: AttackBody) -> SafeJSONResponse:
        return await guarded(runner().set_attack(body.attack, body.attack_params))

    @app.post("/api/detector")
    async def detector(body: DetectorBody) -> SafeJSONResponse:
        return await guarded(runner().set_detector(body.detector))

    @app.get("/api/bench")
    async def bench() -> SafeJSONResponse:
        return SafeJSONResponse(await asyncio.to_thread(read_bench, bench_file))

    # ------------------------------------------------------------------ Tiger Data endpoints

    @app.get("/api/runs")
    async def runs(limit: int = 50) -> SafeJSONResponse:
        return SafeJSONResponse(await app.state.store.list_runs(limit))

    @app.get("/api/runs/{run_id}/timeline")
    async def timeline(run_id: str) -> SafeJSONResponse:
        result = await app.state.store.timeline(run_id)
        if result.get("available") and result.get("run") is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return SafeJSONResponse(result)

    @app.get("/api/analytics")
    async def analytics() -> SafeJSONResponse:
        return SafeJSONResponse(await app.state.store.analytics())

    # ------------------------------------------------------------------ WebSocket feed

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        hub_: Hub = app.state.hub
        r = runner()
        cat = await r.catalog()
        client = hub_.connect()  # connect + hello + scenario happen with no await in between: nothing interleaves
        hub_.send(client, messages.hello(cat, r.status.to_dict()))
        if r.scenario_info is not None:
            hub_.send(client, messages.scenario(r.scenario_info))

        async def pump() -> None:
            while (text := await client.get()) is not None:
                await websocket.send_text(text)

        async def drain() -> None:
            while True:  # inbound messages are ignored; this just notices the disconnect
                await websocket.receive_text()

        try:
            async with anyio.create_task_group() as tg:
                async def until_either_ends(fn: Callable[[], Awaitable[None]]) -> None:
                    try:
                        await fn()
                    except Exception:  # noqa: BLE001 - disconnects surface as various exception types
                        pass
                    finally:
                        tg.cancel_scope.cancel()

                tg.start_soon(until_either_ends, pump)
                tg.start_soon(until_either_ends, drain)
        finally:
            hub_.disconnect(client)
        if websocket.client_state == WebSocketState.CONNECTED:  # the hub dropped a stalled client
            try:
                await websocket.close(code=1013)
            except Exception:  # noqa: BLE001
                pass

    return app


app = create_app()
