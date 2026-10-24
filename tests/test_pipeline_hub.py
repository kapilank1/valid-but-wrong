"""Hub fan-out: bounded per-client queues that drop old ticks, never alerts/status/scenario."""

import asyncio
import json
import math
import time

import numpy as np

from vbw.pipeline.hub import Hub
from vbw.pipeline.jsonsafe import dumps


def tick(i: int) -> dict:
    return {"type": "tick", "i": i}


def drain(client) -> list[dict]:
    async def main():
        out = []
        while len(client):
            out.append(json.loads(await client.get()))
        return out

    return asyncio.run(main())


def test_full_queue_drops_oldest_ticks_but_keeps_alerts_in_order():
    hub = Hub(max_ticks=3)
    client = hub.connect()
    hub.publish({"type": "scenario", "scenario": {"id": "x"}})
    for i in range(5):
        hub.publish(tick(i))
    hub.publish({"type": "alert", "spoken": "Flight, Guidance."})
    for i in range(5, 10):
        hub.publish(tick(i))
    hub.publish({"type": "status", "status": {}})

    got = drain(client)
    kinds = [m["type"] for m in got]
    assert kinds == ["scenario", "alert", "tick", "tick", "tick", "status"]
    assert [m["i"] for m in got if m["type"] == "tick"] == [7, 8, 9], "the newest ticks survive"
    assert client.dropped == 7


def test_publish_to_a_stalled_client_is_fast():
    hub = Hub(max_ticks=64)
    hub.connect()
    payload = {"type": "tick", "verdict": {"score": 0.5, "channels": {c: 1.0 for c in "abcdefgh"}}}
    start = time.perf_counter()
    for _ in range(20_000):
        hub.publish(payload)
    assert time.perf_counter() - start < 2.0


def test_client_that_hoards_undroppable_messages_is_disconnected():
    hub = Hub(max_ticks=4, max_pending=10)
    client = hub.connect()
    for _ in range(11):
        hub.publish({"type": "alert"})
    assert client.closed and client not in hub.clients
    hub.publish({"type": "alert"})  # publishing after the disconnect is harmless


def test_get_wakes_up_on_publish():
    hub = Hub()

    async def main():
        client = hub.connect()
        waiter = asyncio.create_task(client.get())
        await asyncio.sleep(0.01)
        assert not waiter.done()
        hub.publish({"type": "status", "status": {"running": True}})
        return json.loads(await asyncio.wait_for(waiter, 1))

    assert asyncio.run(main())["status"]["running"] is True


def test_get_returns_none_after_close():
    hub = Hub()

    async def main():
        client = hub.connect()
        waiter = asyncio.create_task(client.get())
        await asyncio.sleep(0)
        hub.disconnect(client)
        return await asyncio.wait_for(waiter, 1)

    assert asyncio.run(main()) is None


def test_messages_are_strict_json():
    text = dumps({"a": math.nan, "b": [math.inf, -math.inf, 1.5], "c": np.float32(2.5), "d": np.float64("nan"),
                  "e": np.int64(3), "f": np.array([1.0, np.nan])})
    assert json.loads(text) == {"a": None, "b": [None, None, 1.5], "c": 2.5, "d": None, "e": 3, "f": [1.0, None]}
    assert "NaN" not in text and "Infinity" not in text
