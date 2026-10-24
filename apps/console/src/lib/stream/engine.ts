import { API_BASE } from "@/lib/config";
import { TelemetryBuffers } from "@/lib/stream/buffers";
import { initialStreamState, isTimeRewind, reduceStream, type StreamAction, type StreamState } from "@/lib/stream/reducer";
import { LiveTransport, probeHealth } from "@/lib/transport/live";
import { MockTransport } from "@/lib/transport/mock";
import type { ConnectionInfo, Transport } from "@/lib/transport/types";
import type { AttackRequest, ControlRequest, DetectorRequest, ServerMessage } from "@/lib/types";
import { CalloutQueue } from "@/lib/voice/queue";

/** React sees the stream at most this often; canvases read the buffers every animation frame. */
export const UI_THROTTLE_MS = 200;

export interface ConsoleSnapshot {
  stream: StreamState;
  connection: ConnectionInfo;
}

export const INITIAL_SNAPSHOT: ConsoleSnapshot = {
  stream: initialStreamState,
  connection: { mode: "live", state: "connecting" },
};

export interface EngineOptions {
  forceMock: boolean;
}

/**
 * Owns the connection, the chart buffers and the reduced stream state, outside React.
 * Components subscribe through useSyncExternalStore and get throttled snapshots.
 */
export class ConsoleEngine {
  readonly buffers = new TelemetryBuffers();
  private state: StreamState = initialStreamState;
  private connection: ConnectionInfo = INITIAL_SNAPSHOT.connection;
  private snapshot: ConsoleSnapshot = INITIAL_SNAPSHOT;
  private readonly listeners = new Set<() => void>();
  private transport: Transport | null = null;
  private emitTimer: ReturnType<typeof setTimeout> | null = null;
  private lastEmit = 0;
  private generation = 0;

  constructor(readonly voice: CalloutQueue | null = null) {}

  async start({ forceMock }: EngineOptions): Promise<void> {
    const gen = ++this.generation;
    this.transport?.close();
    this.transport = null;
    this.setConnection({ mode: forceMock ? "mock" : "live", state: "connecting" });

    const live = !forceMock && (await probeHealth());
    if (gen !== this.generation) return; // stopped or restarted while probing
    this.transport = live
      ? new LiveTransport()
      : new MockTransport(forceMock ? "mock mode requested (?mock=1)" : `server unreachable at ${API_BASE}`);
    this.dispatch({ type: "local:log", kind: "link", text: live ? `Connected to ${API_BASE}` : "Mock stream: in-browser generator" });
    this.transport.connect({
      onMessage: (msg) => this.handleMessage(msg),
      onConnection: (info) => this.setConnection(info),
    });
  }

  stop(): void {
    this.generation += 1;
    this.transport?.close();
    this.transport = null;
    if (this.emitTimer) clearTimeout(this.emitTimer);
    this.emitTimer = null;
  }

  get mode(): Transport["mode"] | null {
    return this.transport?.mode ?? null;
  }

  // --------------------------------------------------------------------- store interface

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  getSnapshot = (): ConsoleSnapshot => this.snapshot;

  getServerSnapshot = (): ConsoleSnapshot => INITIAL_SNAPSHOT;

  // --------------------------------------------------------------------- commands

  control(body: ControlRequest): Promise<unknown> {
    return this.command(`control ${body.action}`, (t) => t.control(body));
  }

  setAttack(body: AttackRequest): Promise<unknown> {
    return this.command(`attack ${body.attack}`, (t) => t.setAttack(body));
  }

  setDetector(body: DetectorRequest): Promise<unknown> {
    return this.command(`detector ${body.detector}`, (t) => t.setDetector(body));
  }

  async getBench(): Promise<unknown> {
    if (!this.transport) return null;
    try {
      return await this.transport.getBench();
    } catch {
      return null;
    }
  }

  // --------------------------------------------------------------------- internals

  handleMessage(msg: ServerMessage): void {
    if (msg.type === "scenario") this.buffers.clear();
    if (msg.type === "tick") {
      if (isTimeRewind(this.state.latest, msg)) this.buffers.clear();
      this.buffers.push(msg);
    }
    if (msg.type === "alert") this.voice?.enqueue(msg.spoken);
    this.dispatch(msg);
  }

  private dispatch(action: StreamAction): void {
    this.state = reduceStream(this.state, action);
    // Ticks are throttled; anything rarer (alerts, status, scenario) is shown immediately.
    this.scheduleEmit(action.type !== "tick");
  }

  private async command(label: string, run: (t: Transport) => Promise<unknown>): Promise<unknown> {
    const transport = this.transport;
    if (!transport) {
      this.dispatch({ type: "local:log", kind: "info", text: `Not connected: ${label} ignored` });
      return null;
    }
    try {
      return await run(transport);
    } catch (err) {
      const detail = err instanceof Error ? err.message : String(err);
      this.dispatch({ type: "local:log", kind: "info", text: `Command failed: ${label}`, detail });
      return null;
    }
  }

  private setConnection(info: ConnectionInfo): void {
    const prev = this.connection;
    this.connection = info;
    if (prev.state === "open" && info.state === "reconnecting") {
      this.dispatch({ type: "local:log", kind: "link", text: "Link to server lost, reconnecting" });
    } else if (prev.state === "reconnecting" && info.state === "open") {
      this.dispatch({ type: "local:log", kind: "link", text: "Link to server restored" });
    } else {
      this.scheduleEmit(true);
    }
  }

  private scheduleEmit(urgent: boolean): void {
    if (urgent) {
      this.emit();
      return;
    }
    if (this.emitTimer) return;
    const wait = Math.max(0, UI_THROTTLE_MS - (Date.now() - this.lastEmit));
    this.emitTimer = setTimeout(() => this.emit(), wait);
  }

  private emit(): void {
    if (this.emitTimer) clearTimeout(this.emitTimer);
    this.emitTimer = null;
    this.lastEmit = Date.now();
    this.snapshot = { stream: this.state, connection: this.connection };
    for (const l of this.listeners) l();
  }
}
