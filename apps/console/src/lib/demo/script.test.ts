import { describe, expect, it } from "vitest";
import {
  DEMO_SCENARIO,
  DEMO_SPEED,
  DEMO_TIMING,
  IDLE_DEMO,
  cancelDemo,
  demoView,
  isDemoActive,
  observeDemo,
  startDemo,
  stepDemo,
  type DemoObservation,
  type DemoState,
} from "@/lib/demo/script";
import { initialStreamState } from "@/lib/stream/reducer";
import { makeTick } from "@/lib/stream/testUtils";

const obs = (patch: Partial<DemoObservation> = {}): DemoObservation => ({
  t: 0,
  running: true,
  scenario: DEMO_SCENARIO,
  speed: DEMO_SPEED,
  alert: false,
  lastAlertT: null,
  detector: "base",
  lieKm: 0,
  ppoAvailable: true,
  ...patch,
});

/** Feed observations in order and collect every command the script issues. */
function run(start: DemoState, observations: Partial<DemoObservation>[]) {
  let state = start;
  const commands: string[] = [];
  for (const o of observations) {
    const step = stepDemo(state, obs(o));
    state = step.state;
    for (const c of step.commands) {
      commands.push(c.kind === "control" ? `control:${c.body.action}` : c.kind === "attack" ? `attack:${c.body.attack}` : `detector:${c.body.detector}`);
    }
  }
  return { state, commands };
}

describe("demo script", () => {
  it("starts by resetting SES-9 at 10x on base with no attack, then starting it", () => {
    const { state, commands } = startDemo();
    expect(state.phase).toBe("launching");
    expect(commands.map((c) => c.kind === "control" && c.body.action)).toEqual(["reset", "start"]);
    const reset = commands[0];
    expect(reset?.kind === "control" && reset.body).toMatchObject({ scenario: DEMO_SCENARIO, attack: "none", detector: "base", speed: DEMO_SPEED });
  });

  it("waits for the fresh run before doing anything (stale ticks from the old run don't count)", () => {
    const s = startDemo().state;
    expect(stepDemo(s, obs({ t: 400 })).state.phase).toBe("launching"); // the previous run, still playing
    expect(stepDemo(s, obs({ t: 2, speed: 1 })).state.phase).toBe("launching"); // speed not applied yet
    expect(stepDemo(s, obs({ t: 2 })).state.phase).toBe("cruise");
  });

  it("runs the whole pitch: bias → caught → clear → PPO lies with base silent → harden → caught → stop", () => {
    const { state, commands } = run(startDemo().state, [
      { t: 1 }, // fresh run
      { t: 30 },
      { t: 60 }, // bias on
      { t: 62, alert: true, lastAlertT: 60.6 }, // caught (snapshot lands later than the alert itself)
      { t: 70, alert: true, lastAlertT: 60.6 }, // still alerting: wait
      { t: 90, alert: false }, // cleared, but too early to arm the PPO agent
      { t: 150 }, // arm it: start_t = 190
      { t: 170 },
      { t: 190 }, // it begins
      { t: 250, lieKm: 2.1 },
      { t: 266, lieKm: 3.6 }, // big enough: swap
      { t: 266.5, detector: "hardened", lieKm: 3.6 },
      { t: 270, detector: "hardened", alert: true, lastAlertT: 267.4, lieKm: 3.7 },
    ]);
    expect(commands).toEqual(["attack:bias", "attack:none", "attack:ppo", "detector:hardened", "control:stop"]);
    expect(state.phase).toBe("done");
    expect(state.marks).toMatchObject({ biasAt: 60, biasAlertAt: 60.6, ppoAt: 190, swapAt: 266, swapLieKm: 3.6, hardenAlertAt: 267.4 });
    expect(state.marks.baseAlertDuringPpoAt).toBeUndefined();
    expect(demoView(state, { lieKm: 3.7 }).title).toBe("Hardened caught it at T+267 s");
  });

  it("arms the PPO agent early with a future start_t so the server-side load doesn't eat flight time", () => {
    const clearing: DemoState = { phase: "clearing", since: 65, marks: {}, note: null };
    expect(stepDemo(clearing, obs({ t: DEMO_TIMING.ppoAt - DEMO_TIMING.ppoArmLead - 1 })).commands).toEqual([]);
    const armed = stepDemo(clearing, obs({ t: DEMO_TIMING.ppoAt - DEMO_TIMING.ppoArmLead }));
    expect(armed.state.phase).toBe("armed");
    expect(armed.commands).toEqual([{ kind: "attack", body: { attack: "ppo", attack_params: { intensity: 1, start_t: DEMO_TIMING.ppoAt } } }]);
    // An old alert (e.g. the bias) is never taken as the hardened detector's.
    const harden: DemoState = { phase: "harden", since: 300, marks: { swapAt: 300 }, note: null };
    const done = stepDemo(harden, obs({ t: 305, alert: true, detector: "hardened", lastAlertT: 60.6 }));
    expect(done.state.marks.hardenAlertAt).toBe(305);
  });

  it("swaps at the latest time even if the lie stays small, and notes a base alert during PPO", () => {
    const ppo: DemoState = { phase: "ppo", since: 190, marks: { ppoAt: 190 }, note: null };
    const a = stepDemo(ppo, obs({ t: 250, alert: true, lieKm: 1 }));
    expect(a.state.marks.baseAlertDuringPpoAt).toBe(250);
    const b = stepDemo(a.state, obs({ t: DEMO_TIMING.swapLatestAt, lieKm: 1.2 }));
    expect(b.state.phase).toBe("harden");
    expect(b.commands).toEqual([{ kind: "detector", body: { detector: "hardened" } }]);
  });

  it("moves on when an alert never comes, and stops cleanly if the run ends early", () => {
    const bias: DemoState = { phase: "bias", since: 60, marks: { biasAt: 60 }, note: null };
    const late = stepDemo(bias, obs({ t: 60 + DEMO_TIMING.alertTimeout + 1 }));
    expect(late.state.phase).toBe("clearing");
    expect(late.state.note).toMatch(/not caught/);
    const ended = stepDemo(late.state, obs({ t: 545, running: false }));
    expect(ended.state.phase).toBe("done");
    expect(ended.commands).toEqual([]);
  });

  it("finishes with a note when the server has no PPO policy", () => {
    const clearing: DemoState = { phase: "clearing", since: 90, marks: {}, note: null };
    const step = stepDemo(clearing, obs({ t: 190, ppoAvailable: false }));
    expect(step.state.phase).toBe("done");
    expect(step.state.note).toMatch(/PPO/);
    expect(step.commands).toEqual([{ kind: "control", body: { action: "stop" } }]);
  });

  it("can be cancelled, after which it issues nothing", () => {
    const cancelled = cancelDemo({ phase: "ppo", since: 190, marks: {}, note: null }).state;
    expect(cancelled.phase).toBe("cancelled");
    expect(isDemoActive(cancelled)).toBe(false);
    expect(stepDemo(cancelled, obs({ t: 300, lieKm: 9 })).commands).toEqual([]);
    expect(isDemoActive(IDLE_DEMO)).toBe(false);
  });

  it("observes mission time, alert, detector and the altitude lie from the stream", () => {
    const tick = makeTick(212, { altitude: 150_000 }, { altitude: 152_500 }, { alert: true, detector: "hardened", active: true, attack: "ppo" });
    const status = { running: true, scenario: DEMO_SCENARIO, attack: "ppo", attack_params: {}, detector: "hardened", speed: 10, t: 212, frames: 1, alerts: 1 };
    const catalog = {
      scenarios: [],
      attacks: [{ name: "ppo", label: "PPO", description: "", schemas: ["falcon9"], params: {} }],
      detectors: [],
    };
    const log = [
      { id: 3, t: 212, kind: "info" as const, text: "x" },
      { id: 2, t: 208.4, kind: "alert" as const, text: "rising edge" },
      { id: 1, t: 64.3, kind: "alert" as const, text: "older" },
    ];
    const o = observeDemo({ ...initialStreamState, latest: tick, status, catalog, log });
    expect(o).toEqual({ t: 212, running: true, scenario: DEMO_SCENARIO, speed: 10, alert: true, lastAlertT: 208.4, detector: "hardened", lieKm: 2.5, ppoAvailable: true });
    expect(observeDemo(initialStreamState)).toMatchObject({ t: null, running: false, ppoAvailable: false });
  });

  it("labels the PPO step with the live lie size", () => {
    const view = demoView({ phase: "ppo", since: 190, marks: {}, note: null }, { lieKm: 3.42 });
    expect(view).toMatchObject({ step: 3, title: "PPO agent lying · base silent", detail: "picture off by 3.4 km", active: true });
  });
});
