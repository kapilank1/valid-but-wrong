import { attackAvailability } from "@/lib/catalog";
import { missionTime, type StreamState } from "@/lib/stream/reducer";
import { altitudeLieKm } from "@/lib/truthGap";
import type { AttackRequest, ControlRequest, DetectorRequest } from "@/lib/types";

/**
 * The one-click pitch, as a pure step machine. Each call to stepDemo() looks at what the stream shows now and
 * returns the next state plus the commands to send; the React hook (useDemo) only observes and executes.
 *
 *   1 launch   reset + start SES-9 (held-out launch) at 10x on the base detector, no attack
 *   2 bias     at T+60 s: constant bias at full intensity → base catches it → clear
 *   3 ppo      at T+190 s (above the atmosphere): PPO agent; base stays silent while the lie grows to ~3.5 km.
 *              It is armed ~40 mission-s early with start_t = 190: loading the policy on the server takes a few wall
 *              seconds, which at 10x is a minute of flight. (Clearing the bias leaves short "echo" alerts when the
 *              retracted step leaves the 10 s and 60 s L1 windows; the script waits them out before arming.)
 *   4 harden   hot-swap to the hardened detector → it catches the PPO agent
 *   5 done     stop the run
 *
 * Timings are mission seconds, so the script behaves the same at any playback speed.
 */

export const DEMO_SCENARIO = "falcon9:ses-9";
export const DEMO_SPEED = 10;
export const DEMO_TIMING = {
  biasAt: 60,
  ppoAt: 190,
  ppoArmLead: 40, // send the PPO attack this long before ppoAt, with start_t = ppoAt
  swapLieKm: 3.5, // hot-swap once the operator's altitude is this wrong ...
  swapLatestAt: 300, // ... or at this mission time, whichever comes first
  alertTimeout: 60, // mission seconds to wait for an alert before moving on anyway
  clearTimeout: 30, // mission seconds to wait for the bias alert to clear
  freshRunMaxT: 30, // a run counts as the new one only while t is below this
} as const;

export type DemoPhase =
  | "idle"
  | "launching" // commands sent, waiting to see the new run
  | "cruise" // nominal flight before the bias
  | "bias" // bias on, waiting for the alert
  | "clearing" // bias off, waiting for the alert to clear
  | "armed" // PPO sent with a future start_t, waiting for it to begin
  | "ppo" // PPO on, base should stay silent
  | "harden" // hardened swapped in, waiting for its alert
  | "done"
  | "cancelled";

export interface DemoObservation {
  t: number | null; // mission time of the latest tick (or status)
  running: boolean;
  scenario: string | null;
  speed: number | null;
  alert: boolean; // the latest verdict is alerting
  lastAlertT: number | null; // mission time of the most recent alert's rising edge (exact, unlike t)
  detector: string; // detector behind the latest verdict
  lieKm: number | null; // reported − true altitude, km
  ppoAvailable: boolean;
}

export interface DemoMarks {
  biasAt?: number;
  biasAlertAt?: number;
  ppoAt?: number;
  baseAlertDuringPpoAt?: number;
  swapAt?: number;
  swapLieKm?: number;
  hardenAlertAt?: number;
  endedAt?: number;
}

export interface DemoState {
  phase: DemoPhase;
  since: number | null; // mission time the current phase started
  marks: DemoMarks;
  note: string | null; // anything that didn't go to plan, shown to the operator
}

export type DemoCommand =
  | { kind: "control"; body: ControlRequest }
  | { kind: "attack"; body: AttackRequest }
  | { kind: "detector"; body: DetectorRequest };

export interface DemoStep {
  state: DemoState;
  commands: DemoCommand[];
}

export const IDLE_DEMO: DemoState = { phase: "idle", since: null, marks: {}, note: null };

/** What the script needs to know, read off the (throttled) stream state. */
export function observeDemo(stream: StreamState): DemoObservation {
  const verdict = stream.latest?.verdict;
  return {
    t: missionTime(stream),
    running: stream.status?.running ?? false,
    scenario: stream.status?.scenario ?? null,
    speed: stream.status?.speed ?? null,
    alert: verdict?.alert ?? false,
    // The newest "alert" log entry is the rising edge; lastAlert.t keeps moving while the alert is sustained.
    lastAlertT: stream.log.find((e) => e.kind === "alert")?.t ?? null,
    detector: verdict?.detector ?? stream.status?.detector ?? "base",
    lieKm: altitudeLieKm(stream.latest),
    ppoAvailable: attackAvailability(stream.catalog, "ppo", "falcon9").available,
  };
}

const attack = (name: string, t: number): DemoCommand => ({
  kind: "attack",
  body: { attack: name, attack_params: { intensity: 1, start_t: Math.max(0, Math.round(t * 100) / 100) } },
});

export function startDemo(): DemoStep {
  return {
    state: { phase: "launching", since: null, marks: {}, note: null },
    commands: [
      {
        kind: "control",
        body: {
          action: "reset",
          scenario: DEMO_SCENARIO,
          attack: "none",
          attack_params: { intensity: 1, start_t: 0 },
          detector: "base",
          speed: DEMO_SPEED,
        },
      },
      { kind: "control", body: { action: "start", scenario: DEMO_SCENARIO, speed: DEMO_SPEED } },
    ],
  };
}

export function cancelDemo(state: DemoState): DemoStep {
  if (!isDemoActive(state)) return { state, commands: [] };
  return { state: { ...state, phase: "cancelled", note: "Cancelled by operator" }, commands: [] };
}

export function isDemoActive(state: DemoState): boolean {
  return state.phase !== "idle" && state.phase !== "done" && state.phase !== "cancelled";
}

export function stepDemo(state: DemoState, obs: DemoObservation): DemoStep {
  const t = obs.t;
  const to = (phase: DemoPhase, patch: Partial<DemoState> = {}): DemoState => ({ ...state, phase, since: t, ...patch });
  const finish = (marks: DemoMarks, note: string | null = state.note): DemoStep => ({
    state: { phase: "done", since: t, marks: { ...marks, endedAt: t ?? undefined }, note },
    commands: [{ kind: "control", body: { action: "stop" } }],
  });

  if (!isDemoActive(state) || t === null) return { state, commands: [] };
  // The alert's own timestamp when it is newer than `after`; the sampled time otherwise (snapshots are ~5 Hz).
  const alertTime = (after: number | undefined) =>
    obs.lastAlertT !== null && (after === undefined || obs.lastAlertT >= after) ? obs.lastAlertT : t;

  if (state.phase === "launching") {
    const fresh = obs.running && obs.scenario === DEMO_SCENARIO && obs.speed === DEMO_SPEED && t < DEMO_TIMING.freshRunMaxT;
    return { state: fresh ? to("cruise") : state, commands: [] };
  }

  // The replay ended (or someone stopped it) before the script did.
  if (!obs.running) return { state: { ...to("done"), note: state.note ?? "Run stopped before the script finished" }, commands: [] };

  const since = state.since ?? t;
  const marks = state.marks;
  switch (state.phase) {
    case "cruise":
      if (t < DEMO_TIMING.biasAt) return { state, commands: [] };
      return { state: to("bias", { marks: { ...marks, biasAt: t } }), commands: [attack("bias", t)] };

    case "bias":
      if (obs.alert) {
        return { state: to("clearing", { marks: { ...marks, biasAlertAt: alertTime(marks.biasAt) } }), commands: [attack("none", t)] };
      }
      if (t - since > DEMO_TIMING.alertTimeout) {
        return { state: to("clearing", { note: "Bias was not caught in time" }), commands: [attack("none", t)] };
      }
      return { state, commands: [] };

    case "clearing": {
      const cleared = !obs.alert || t - since > DEMO_TIMING.clearTimeout;
      if (!cleared || t < DEMO_TIMING.ppoAt - DEMO_TIMING.ppoArmLead) return { state, commands: [] };
      if (!obs.ppoAvailable) return finish(marks, "PPO policy not available on the server");
      const begins = Math.max(DEMO_TIMING.ppoAt, t + 5);
      return { state: to("armed", { marks: { ...marks, ppoAt: begins } }), commands: [attack("ppo", begins)] };
    }

    case "armed":
      return { state: t >= (marks.ppoAt ?? DEMO_TIMING.ppoAt) ? to("ppo") : state, commands: [] };

    case "ppo": {
      const nextMarks = obs.alert && marks.baseAlertDuringPpoAt === undefined ? { ...marks, baseAlertDuringPpoAt: t } : marks;
      const lie = Math.abs(obs.lieKm ?? 0);
      if (lie < DEMO_TIMING.swapLieKm && t < DEMO_TIMING.swapLatestAt) {
        return { state: nextMarks === marks ? state : { ...state, marks: nextMarks }, commands: [] };
      }
      return {
        state: to("harden", { marks: { ...nextMarks, swapAt: t, swapLieKm: obs.lieKm ?? undefined } }),
        commands: [{ kind: "detector", body: { detector: "hardened" } }],
      };
    }

    case "harden":
      if (obs.alert && obs.detector === "hardened") return finish({ ...marks, hardenAlertAt: alertTime(marks.swapAt) });
      if (t - since > DEMO_TIMING.alertTimeout) return finish(marks, "Hardened detector did not alert in time");
      return { state, commands: [] };

    default:
      return { state, commands: [] };
  }
}

// --------------------------------------------------------------------------- what the operator sees

export const DEMO_STEPS = 5;

export interface DemoView {
  step: number; // 1..5, 0 when idle
  title: string;
  detail: string;
  active: boolean;
}

const fmtT = (t: number | undefined) => (t === undefined ? "—" : `T+${Math.round(t)} s`);

export function demoView(state: DemoState, obs: Pick<DemoObservation, "lieKm">): DemoView {
  const m = state.marks;
  const lie = obs.lieKm === null ? "—" : `${Math.abs(obs.lieKm).toFixed(1)} km`;
  const active = isDemoActive(state);
  switch (state.phase) {
    case "idle":
      return { step: 0, title: "Demo", detail: "", active };
    case "launching":
      return { step: 1, title: "Launching SES-9 at 10×", detail: "held-out flight · base detector", active };
    case "cruise":
      return { step: 1, title: "Nominal flight", detail: `bias at T+${DEMO_TIMING.biasAt} s`, active };
    case "bias":
      return { step: 2, title: "Bias attack · L1 bait", detail: "valid CRC, impossible altitude", active };
    case "clearing":
      return {
        step: 2,
        title: m.biasAlertAt !== undefined ? `Bias caught at ${fmtT(m.biasAlertAt)}` : "Bias cleared",
        detail: `PPO agent at T+${DEMO_TIMING.ppoAt} s`,
        active,
      };
    case "armed":
      return { step: 3, title: "PPO agent armed", detail: `lies from ${fmtT(m.ppoAt)}`, active };
    case "ppo":
      return {
        step: 3,
        title: m.baseAlertDuringPpoAt !== undefined ? "PPO agent lying · base alerted" : "PPO agent lying · base silent",
        detail: `picture off by ${lie}`,
        active,
      };
    case "harden":
      return { step: 4, title: "Hot-swap → hardened", detail: `lie was ${m.swapLieKm === undefined ? "—" : `${Math.abs(m.swapLieKm).toFixed(1)} km`}`, active };
    case "done":
      return {
        step: 5,
        title: m.hardenAlertAt !== undefined ? `Hardened caught it at ${fmtT(m.hardenAlertAt)}` : "Demo finished",
        detail: state.note ?? "",
        active,
      };
    case "cancelled":
      return { step: 0, title: "Demo cancelled", detail: "", active };
  }
}
