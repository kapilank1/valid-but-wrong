// Mirrors vbw/schema.py (dataclass to_dict() output) and docs/CONTRACT.md.
// Keep field names identical to the Python side; change both together.

export type SchemaName = "falcon9" | "quad" | "spacecraft";

export const SCHEMA_IDS: Record<number, SchemaName> = {
  1: "falcon9",
  2: "quad",
  3: "spacecraft",
};

export interface ChannelSchema {
  schema_id: number;
  name: SchemaName;
  channels: readonly string[];
  units: readonly string[];
}

export const FALCON9: ChannelSchema = {
  schema_id: 1,
  name: "falcon9",
  channels: ["speed", "altitude", "vz", "vx", "downrange", "accel", "angle", "q"],
  units: ["m/s", "m", "m/s", "m/s", "m", "m/s^2", "deg", "Pa"],
};

export const QUAD: ChannelSchema = {
  schema_id: 2,
  name: "quad",
  channels: [
    "pn", "pe", "alt",
    "vn", "ve", "vu",
    "roll", "pitch", "yaw",
    "p", "q", "r",
    "ax", "ay", "az",
    "baro_alt", "groundspeed", "battery_v",
  ],
  units: [
    "m", "m", "m",
    "m/s", "m/s", "m/s",
    "rad", "rad", "rad",
    "rad/s", "rad/s", "rad/s",
    "m/s^2", "m/s^2", "m/s^2",
    "m", "m/s", "V",
  ],
};

export const SPACECRAFT: ChannelSchema = {
  schema_id: 3,
  name: "spacecraft",
  channels: ["value"],
  units: ["norm"],
};

export const SCHEMAS: Record<SchemaName, ChannelSchema> = {
  falcon9: FALCON9,
  quad: QUAD,
  spacecraft: SPACECRAFT,
};

// --------------------------------------------------------------------------- frames & verdicts

export interface Frame {
  vehicle_id: string;
  schema_id: number;
  seq: number; // uint16, wraps at 65536
  t: number; // vehicle mission time, seconds
  channels: Record<string, number>;
}

export interface L0Result {
  sync_ok: boolean;
  crc_ok: boolean;
  seq_ok: boolean;
  time_ok: boolean;
  reasons: string[];
  ok: boolean; // added by Verdict.to_dict()
}

export interface Violation {
  rule: string;
  channels: string[];
  value: number;
  threshold: number;
  message: string;
}

export interface Verdict {
  vehicle_id: string;
  seq: number;
  t: number;
  l0: L0Result;
  l1_violations: Violation[];
  l1_score: number; // 0..1
  l2_residuals: Record<string, number>; // per-channel robust z-score
  l2_score: number; // 0..1
  l3_prob: number | null; // hardened discriminator P(forged)
  score: number; // combined 0..1
  alert: boolean;
  explanation: string;
  detector: string; // "base" | "hardened"
}

// --------------------------------------------------------------------------- catalog & status

export interface ScenarioInfo {
  id: string;
  label: string;
  schema: string | number;
  vehicle_id: string;
  rate_hz: number;
  duration_s: number | null;
  // Named mission events (seconds), e.g. {maxq: 81, meco: 161}. Spacecraft scenarios may also carry
  // labeled anomaly ranges; see lib/scenario.ts for the shapes we accept.
  events: Record<string, unknown> | null;
  description: string;
  anomalies?: unknown;
}

export type AttackParams = Record<string, number | string | boolean>;

export interface AttackInfo {
  name: string;
  label: string;
  description: string;
  schemas: string[];
  params: AttackParams;
  available?: boolean; // optional: false when e.g. the PPO policy file is missing
}

export interface DetectorInfo {
  name: string;
  label: string;
  description: string;
  available?: boolean;
  threshold?: number; // optional: alert threshold on the combined score (console defaults to 0.5)
  thresholds?: Partial<Record<string, number>>; // optional: calibrated threshold per schema name
}

export interface Catalog {
  scenarios: ScenarioInfo[];
  attacks: AttackInfo[];
  detectors: DetectorInfo[];
}

export interface Status {
  running: boolean;
  scenario: string | null;
  attack: string;
  attack_params: AttackParams;
  detector: string;
  speed: number;
  t: number;
  frames: number;
  alerts: number;
}

// --------------------------------------------------------------------------- REST bodies

export type ControlAction = "start" | "stop" | "reset";

export interface ControlRequest {
  action: ControlAction;
  scenario?: string;
  attack?: string;
  attack_params?: AttackParams;
  detector?: string;
  speed?: number;
}

export interface AttackRequest {
  attack: string;
  attack_params?: AttackParams;
}

export interface DetectorRequest {
  detector: string;
}

// --------------------------------------------------------------------------- WebSocket messages

export interface TickAttack {
  name: string;
  active: boolean;
}

export type ServerMessage =
  | { type: "hello"; catalog: Catalog; status: Status }
  | { type: "scenario"; scenario: ScenarioInfo }
  | { type: "tick"; truth: Frame; reported: Frame; verdict: Verdict; attack: TickAttack }
  | { type: "status"; status: Status }
  | { type: "alert"; verdict: Verdict; spoken: string };

export type TickMessage = Extract<ServerMessage, { type: "tick" }>;
