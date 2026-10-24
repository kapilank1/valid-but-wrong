import type { Catalog, ControlRequest, Status } from "@/lib/types";

/** What the operator has dialed in on the top bar; sent with Start/Reset. */
export interface RunControls {
  scenario: string;
  attack: string;
  intensity: number; // 0..1
  startT: number; // mission seconds
  detector: string;
  speed: number;
}

export const SPEEDS = [0.5, 1, 2, 5, 10] as const;

function numberParam(v: unknown, fallback: number): number {
  const n = typeof v === "string" ? Number(v) : v;
  return typeof n === "number" && Number.isFinite(n) ? n : fallback;
}

export function initialControls(status: Status, catalog: Catalog): RunControls {
  return {
    scenario: status.scenario ?? catalog.scenarios[0]?.id ?? "",
    attack: status.attack || "none",
    intensity: numberParam(status.attack_params.intensity, 0.5),
    startT: numberParam(status.attack_params.start_t, 60),
    detector: status.detector || "base",
    speed: status.speed > 0 ? status.speed : 1,
  };
}

export function attackParams(c: Pick<RunControls, "intensity" | "startT">) {
  return { intensity: round2(c.intensity), start_t: Math.max(0, round2(c.startT)) };
}

export function startRequest(c: RunControls): ControlRequest {
  return {
    action: "start",
    scenario: c.scenario || undefined,
    attack: c.attack,
    attack_params: attackParams(c),
    detector: c.detector,
    speed: c.speed,
  };
}

export function resetRequest(c: RunControls): ControlRequest {
  return { ...startRequest(c), action: "reset" };
}

function round2(v: number): number {
  return Math.round(v * 100) / 100;
}
