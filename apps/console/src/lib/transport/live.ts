import { API_BASE, HEALTH_TIMEOUT_MS, wsUrl } from "@/lib/config";
import { backoffDelay } from "@/lib/transport/backoff";
import { parseServerMessage } from "@/lib/transport/parse";
import type { Transport, TransportHandlers } from "@/lib/transport/types";
import type { AttackRequest, ControlRequest, DetectorRequest, Status } from "@/lib/types";

export async function probeHealth(base: string = API_BASE, timeoutMs = HEALTH_TIMEOUT_MS): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${base}/api/health`, { signal: controller.signal, cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/** REST + WebSocket client for the pipeline server, reconnecting with exponential backoff. */
export class LiveTransport implements Transport {
  readonly mode = "live" as const;
  private ws: WebSocket | null = null;
  private handlers: TransportHandlers | null = null;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private attempt = 0;
  private everOpened = false;
  private closed = false;

  constructor(private readonly base: string = API_BASE) {}

  connect(handlers: TransportHandlers): void {
    this.handlers = handlers;
    this.closed = false;
    this.open();
  }

  close(): void {
    this.closed = true;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = null;
    this.ws?.close();
    this.ws = null;
    this.handlers?.onConnection({ mode: "live", state: "closed" });
  }

  control(body: ControlRequest): Promise<Status> {
    return this.post<Status>("/api/control", body);
  }

  setAttack(body: AttackRequest): Promise<unknown> {
    return this.post("/api/attack", body);
  }

  setDetector(body: DetectorRequest): Promise<unknown> {
    return this.post("/api/detector", body);
  }

  async getBench(): Promise<unknown> {
    const res = await fetch(`${this.base}/api/bench`, { cache: "no-store" });
    if (!res.ok) throw new Error(`GET /api/bench → ${res.status}`);
    return res.json();
  }

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`POST ${path} → ${res.status} ${detail.slice(0, 200)}`);
    }
    return (await res.json()) as T;
  }

  private open(): void {
    if (this.closed || !this.handlers) return;
    const handlers = this.handlers;
    handlers.onConnection({ mode: "live", state: this.everOpened ? "reconnecting" : "connecting" });

    let ws: WebSocket;
    try {
      ws = new WebSocket(wsUrl(this.base));
    } catch {
      this.scheduleReconnect();
      return;
    }
    this.ws = ws;
    ws.onopen = () => {
      this.attempt = 0;
      this.everOpened = true;
      handlers.onConnection({ mode: "live", state: "open" });
    };
    ws.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== "string") return;
      const msg = parseServerMessage(event.data);
      if (msg) handlers.onMessage(msg);
    };
    // onerror is always followed by onclose, which owns the reconnect.
    ws.onclose = () => {
      if (this.ws === ws) this.ws = null;
      if (!this.closed) this.scheduleReconnect();
    };
  }

  private scheduleReconnect(): void {
    const delay = backoffDelay(this.attempt);
    this.attempt += 1;
    this.handlers?.onConnection({
      mode: "live",
      state: "reconnecting",
      reason: "link to pipeline server lost",
      retryInMs: delay,
    });
    this.retryTimer = setTimeout(() => this.open(), delay);
  }
}
