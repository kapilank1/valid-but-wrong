"""WebSocket fan-out. `publish` is synchronous and never waits on a client.

Each client has a bounded outbox. When it is full, the oldest *tick* is dropped (ticks are a firehose; only
the newest matter). alert/status/scenario/hello messages are never dropped; a client that lets those pile
up past `max_pending` is considered dead and is disconnected instead of growing memory without bound.
Messages are serialized once per publish, not once per client.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

from vbw.pipeline.jsonsafe import dumps
from vbw.pipeline.messages import DROPPABLE_TYPES

log = logging.getLogger("vbw.pipeline")


@dataclass
class _Item:
    text: str
    droppable: bool


class Client:
    """One subscriber's outbox. The WebSocket endpoint drains it with `await client.get()`."""

    def __init__(self, max_ticks: int = 256, max_pending: int = 1024):
        self.max_ticks = max_ticks
        self.max_pending = max_pending
        self.dropped = 0
        self.closed = False
        self._items: deque[_Item] = deque()
        self._ticks = 0
        self._ready = asyncio.Event()

    def __len__(self) -> int:
        return len(self._items)

    def push(self, text: str, droppable: bool) -> None:
        if self.closed:
            return
        if droppable:
            if self._ticks >= self.max_ticks:
                self._drop_oldest_tick()
            self._ticks += 1
        elif len(self._items) - self._ticks >= self.max_pending:
            log.warning("ws client stalled with %d undelivered messages; disconnecting", self.max_pending)
            self.close()
            return
        self._items.append(_Item(text, droppable))
        self._ready.set()

    def _drop_oldest_tick(self) -> None:
        for i, item in enumerate(self._items):
            if item.droppable:
                del self._items[i]
                self._ticks -= 1
                self.dropped += 1
                return

    async def get(self) -> str | None:
        """Next message text, or None once the client is closed."""
        while not self._items:
            if self.closed:
                return None
            self._ready.clear()
            await self._ready.wait()
        item = self._items.popleft()
        if item.droppable:
            self._ticks -= 1
        return item.text

    def close(self) -> None:
        self.closed = True
        self._items.clear()
        self._ticks = 0
        self._ready.set()


class Hub:
    def __init__(self, max_ticks: int = 256, max_pending: int = 1024):
        self.max_ticks = max_ticks
        self.max_pending = max_pending
        self.clients: set[Client] = set()
        self.published = 0

    def connect(self) -> Client:
        client = Client(self.max_ticks, self.max_pending)
        self.clients.add(client)
        return client

    def disconnect(self, client: Client) -> None:
        client.close()
        self.clients.discard(client)

    def send(self, client: Client, message: dict[str, Any]) -> None:
        """Queue a message for one client (e.g. the hello on connect)."""
        client.push(dumps(message), message.get("type") in DROPPABLE_TYPES)

    def publish(self, message: dict[str, Any]) -> None:
        """Queue a message for every client. Never blocks; safe to call from the runner's hot loop."""
        self.published += 1
        if not self.clients:
            return
        text = dumps(message)
        droppable = message.get("type") in DROPPABLE_TYPES
        for client in list(self.clients):
            client.push(text, droppable)
            if client.closed:
                self.clients.discard(client)

    def stats(self) -> dict[str, Any]:
        return {
            "clients": len(self.clients),
            "published": self.published,
            "dropped": sum(c.dropped for c in self.clients),
        }
