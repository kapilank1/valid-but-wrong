import type {
  Catalog,
  ScenarioInfo,
  ServerMessage,
  Status,
  TickMessage,
  Violation,
} from "@/lib/types";

/**
 * Pure reducer for everything the text panels show. Chart history lives in TelemetryBuffers;
 * this state is small and safe to snapshot into React at ~5 Hz.
 */

export interface LinkCounters {
  frames: number;
  sync: number;
  crc: number;
  seq: number;
  time: number;
  allOk: number;
  forged: number; // ticks where the attacker rewrote the frame (god view)
  forgedPassing: number; // forged frames that still passed every L0 check
}

export type LogKind = "scenario" | "attack" | "alert" | "clear" | "detector" | "link" | "info";

export interface LogEntry {
  id: number;
  t: number;
  kind: LogKind;
  text: string;
  detail?: string;
}

export interface AlertInfo {
  t: number;
  spoken: string;
  explanation: string;
  score: number;
  detector: string;
  violations: Violation[];
}

export interface StreamState {
  catalog: Catalog | null;
  status: Status | null;
  scenario: ScenarioInfo | null;
  latest: TickMessage | null;
  link: LinkCounters;
  lastAlert: AlertInfo | null;
  alertCount: number;
  log: LogEntry[]; // newest first
  nextLogId: number;
}

export type LocalAction =
  | { type: "local:log"; kind: LogKind; text: string; detail?: string }
  | { type: "local:reset" };

export type StreamAction = ServerMessage | LocalAction;

export const LOG_LIMIT = 200;
/** Keep the alert banner up this long (mission seconds) after the verdict stops alerting. */
export const ALERT_HOLD_S = 4;

export const EMPTY_LINK: LinkCounters = {
  frames: 0,
  sync: 0,
  crc: 0,
  seq: 0,
  time: 0,
  allOk: 0,
  forged: 0,
  forgedPassing: 0,
};

export const initialStreamState: StreamState = {
  catalog: null,
  status: null,
  scenario: null,
  latest: null,
  link: EMPTY_LINK,
  lastAlert: null,
  alertCount: 0,
  log: [],
  nextLogId: 1,
};

/** True when mission time jumped backwards, i.e. the server restarted the replay without telling us. */
export function isTimeRewind(prev: TickMessage | null, next: TickMessage): boolean {
  return prev !== null && next.truth.t < prev.truth.t - 0.5;
}

export function missionTime(state: StreamState): number | null {
  return state.latest?.truth.t ?? state.status?.t ?? null;
}

export function reduceStream(state: StreamState, action: StreamAction): StreamState {
  switch (action.type) {
    case "hello":
      return {
        ...state,
        catalog: action.catalog,
        status: action.status,
        scenario: state.scenario ?? findScenario(action.catalog, action.status.scenario),
      };
    case "scenario":
      return addLog(
        { ...resetRun(state), scenario: action.scenario },
        "scenario",
        `Scenario loaded: ${action.scenario.label}`,
        action.scenario.description || undefined,
      );
    case "status":
      return reduceStatus(state, action.status);
    case "tick":
      return reduceTick(state, action);
    case "alert":
      return reduceAlert(state, action.verdict, action.spoken);
    case "local:log":
      return addLog(state, action.kind, action.text, action.detail);
    case "local:reset":
      return resetRun(state);
    default:
      return state;
  }
}

function resetRun(state: StreamState): StreamState {
  return { ...state, latest: null, link: EMPTY_LINK, lastAlert: null, alertCount: 0 };
}

function findScenario(catalog: Catalog | null, id: string | null): ScenarioInfo | null {
  if (!catalog || !id) return null;
  return catalog.scenarios.find((s) => s.id === id) ?? null;
}

function reduceStatus(state: StreamState, status: Status): StreamState {
  let next: StreamState = { ...state, status };
  const prev = state.status;
  if (prev && prev.detector !== status.detector) {
    next = addLog(next, "detector", `Detector hot-swapped → ${status.detector.toUpperCase()}`);
  }
  if (status.scenario && status.scenario !== state.scenario?.id) {
    const scenario = findScenario(state.catalog, status.scenario);
    if (scenario) next = { ...next, scenario };
  }
  return next;
}

function reduceTick(state: StreamState, tick: TickMessage): StreamState {
  const base = isTimeRewind(state.latest, tick) ? resetRun(state) : state;
  const prev = base.latest;
  const { l0 } = tick.verdict;
  const forged = tick.attack.active;
  const link: LinkCounters = {
    frames: base.link.frames + 1,
    sync: base.link.sync + (l0.sync_ok ? 1 : 0),
    crc: base.link.crc + (l0.crc_ok ? 1 : 0),
    seq: base.link.seq + (l0.seq_ok ? 1 : 0),
    time: base.link.time + (l0.time_ok ? 1 : 0),
    allOk: base.link.allOk + (l0.ok ? 1 : 0),
    forged: base.link.forged + (forged ? 1 : 0),
    forgedPassing: base.link.forgedPassing + (forged && l0.ok ? 1 : 0),
  };
  let next: StreamState = { ...base, latest: tick, link };

  const wasForged = prev?.attack.active ?? false;
  const prevAttack = prev?.attack.name;
  if (forged && (!wasForged || prevAttack !== tick.attack.name)) {
    next = addLog(
      next,
      "attack",
      `Attacker rewriting frames: ${tick.attack.name.toUpperCase()}`,
      "God view: forged frames re-encoded with valid CRC",
      tick.truth.t,
    );
  } else if (!forged && wasForged) {
    next = addLog(next, "attack", "Attacker off the link", undefined, tick.truth.t);
  }

  if (tick.verdict.alert && next.lastAlert) {
    // Keep the banner's explanation current while the alert is sustained.
    next = {
      ...next,
      lastAlert: {
        ...next.lastAlert,
        t: tick.verdict.t,
        explanation: tick.verdict.explanation || next.lastAlert.explanation,
        score: tick.verdict.score,
        detector: tick.verdict.detector,
        violations: tick.verdict.l1_violations,
      },
    };
  }
  if (!tick.verdict.alert && prev?.verdict.alert) {
    next = addLog(next, "clear", "Verdict back to nominal", undefined, tick.truth.t);
  }
  return next;
}

function reduceAlert(state: StreamState, verdict: TickMessage["verdict"], spoken: string): StreamState {
  const lastAlert: AlertInfo = {
    t: verdict.t,
    spoken,
    explanation: verdict.explanation,
    score: verdict.score,
    detector: verdict.detector,
    violations: verdict.l1_violations,
  };
  return addLog(
    { ...state, lastAlert, alertCount: state.alertCount + 1 },
    "alert",
    spoken || "Physics inconsistency detected",
    verdict.explanation || undefined,
    verdict.t,
  );
}

function addLog(state: StreamState, kind: LogKind, text: string, detail?: string, t?: number): StreamState {
  const entry: LogEntry = { id: state.nextLogId, t: t ?? missionTime(state) ?? 0, kind, text, detail };
  return { ...state, log: [entry, ...state.log].slice(0, LOG_LIMIT), nextLogId: state.nextLogId + 1 };
}

// --------------------------------------------------------------------------- selectors

export interface AlertBannerView {
  active: boolean;
  /** True while the latest verdict itself is alerting (vs. holding after it cleared). */
  live: boolean;
  alert: AlertInfo | null;
}

export function selectAlertBanner(state: StreamState): AlertBannerView {
  const verdict = state.latest?.verdict;
  const live = verdict?.alert ?? false;
  const t = missionTime(state);
  const held = state.lastAlert !== null && t !== null && t - state.lastAlert.t <= ALERT_HOLD_S;
  // The tick can land before its "alert" message; show the verdict itself until the callout arrives.
  const alert =
    state.lastAlert ??
    (live && verdict
      ? {
          t: verdict.t,
          spoken: "",
          explanation: verdict.explanation,
          score: verdict.score,
          detector: verdict.detector,
          violations: verdict.l1_violations,
        }
      : null);
  return { active: live || held, live, alert };
}

export function crcPassRate(link: LinkCounters): number {
  return link.frames === 0 ? 1 : link.crc / link.frames;
}
