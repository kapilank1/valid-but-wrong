import type {
  AttackRequest,
  ControlRequest,
  DetectorRequest,
  ServerMessage,
  Status,
} from "@/lib/types";

export type TransportMode = "live" | "mock";

export type ConnectionState = "connecting" | "open" | "reconnecting" | "closed";

export interface ConnectionInfo {
  mode: TransportMode;
  state: ConnectionState;
  /** Why we are in this state, e.g. "server unreachable — using mock stream". */
  reason?: string;
  /** Milliseconds until the next reconnect attempt, when reconnecting. */
  retryInMs?: number;
}

export interface TransportHandlers {
  onMessage(msg: ServerMessage): void;
  onConnection(info: ConnectionInfo): void;
}

/** Same surface for the real pipeline server and the in-browser mock. */
export interface Transport {
  readonly mode: TransportMode;
  connect(handlers: TransportHandlers): void;
  close(): void;
  control(body: ControlRequest): Promise<Status>;
  setAttack(body: AttackRequest): Promise<unknown>;
  setDetector(body: DetectorRequest): Promise<unknown>;
  getBench(): Promise<unknown>;
}
