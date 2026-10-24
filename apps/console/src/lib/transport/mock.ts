import { MOCK_BENCH } from "@/lib/mock/catalog";
import { MockSim } from "@/lib/mock/sim";
import type { Transport, TransportHandlers } from "@/lib/transport/types";
import type { AttackRequest, ControlRequest, DetectorRequest, ServerMessage, Status } from "@/lib/types";

const TICK_MS = 50;

/** Drives MockSim from the wall clock and delivers its messages as if they came over the WebSocket. */
export class MockTransport implements Transport {
  readonly mode = "mock" as const;
  private readonly sim = new MockSim();
  private handlers: TransportHandlers | null = null;
  private timer: ReturnType<typeof setInterval> | null = null;
  private lastWall = 0;

  constructor(private readonly reason: string) {}

  connect(handlers: TransportHandlers): void {
    this.handlers = handlers;
    handlers.onConnection({ mode: "mock", state: "open", reason: this.reason });
    this.deliver(this.sim.hello());
    this.lastWall = performance.now();
    this.timer = setInterval(() => {
      const now = performance.now();
      // Clamp so a backgrounded tab doesn't fast-forward the mission when it wakes up.
      const dt = Math.min(0.25, (now - this.lastWall) / 1000);
      this.lastWall = now;
      this.deliver(this.sim.advance(dt));
    }, TICK_MS);
  }

  close(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
    this.handlers?.onConnection({ mode: "mock", state: "closed" });
  }

  async control(body: ControlRequest): Promise<Status> {
    this.deliver(this.sim.control(body));
    return this.sim.getStatus();
  }

  async setAttack(body: AttackRequest): Promise<unknown> {
    this.deliver(this.sim.setAttack(body));
    return { ok: true };
  }

  async setDetector(body: DetectorRequest): Promise<unknown> {
    this.deliver(this.sim.setDetector(body));
    return { ok: true };
  }

  async getBench(): Promise<unknown> {
    return MOCK_BENCH;
  }

  private deliver(messages: ServerMessage[]): void {
    for (const msg of messages) this.handlers?.onMessage(msg);
  }
}
