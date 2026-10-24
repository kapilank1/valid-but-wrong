import { describe, expect, it } from "vitest";
import { MockSim } from "@/lib/mock/sim";
import type { ServerMessage, TickMessage } from "@/lib/types";

type Run = { ticks: TickMessage[]; alerts: Extract<ServerMessage, { type: "alert" }>[] };

/** Play a scenario for `seconds` of mission time at 1x and collect what the console would receive. */
function run(opts: { scenario?: string; attack?: string; detector?: string; startT?: number; seconds: number; intensity?: number }): Run {
  const sim = new MockSim(42);
  const msgs: ServerMessage[] = [];
  msgs.push(
    ...sim.control({
      action: "start",
      scenario: opts.scenario ?? "falcon9:ses-9",
      attack: opts.attack ?? "none",
      attack_params: { start_t: opts.startT ?? 60, intensity: opts.intensity ?? 0.5 },
      detector: opts.detector ?? "base",
      speed: 1,
    }),
  );
  for (let i = 0; i < opts.seconds * 20; i++) msgs.push(...sim.advance(0.05));
  return {
    ticks: msgs.filter((m): m is TickMessage => m.type === "tick"),
    alerts: msgs.filter((m): m is Run["alerts"][number] => m.type === "alert"),
  };
}

const firstAlertT = (r: Run) => r.alerts[0]?.verdict.t ?? Number.POSITIVE_INFINITY;

describe("MockSim", () => {
  it("emits frames at the scenario rate with increasing seq and time", () => {
    const { ticks } = run({ seconds: 2 });
    expect(ticks.length).toBeGreaterThanOrEqual(19);
    expect(ticks.length).toBeLessThanOrEqual(21);
    for (let i = 1; i < ticks.length; i++) {
      expect(ticks[i]!.truth.seq).toBe((ticks[i - 1]!.truth.seq + 1) & 0xffff);
      expect(ticks[i]!.truth.t).toBeGreaterThan(ticks[i - 1]!.truth.t);
    }
  });

  it("raises no alerts on a clean Falcon 9 ascent", () => {
    const r = run({ seconds: 540 });
    expect(r.alerts).toHaveLength(0);
    expect(r.ticks.every((t) => !t.attack.active)).toBe(true);
    expect(r.ticks.every((t) => t.reported.channels.altitude === t.truth.channels.altitude)).toBe(true);
  });

  it("keeps every forged frame link-valid (L0 all green)", () => {
    const r = run({ attack: "bias", seconds: 90 });
    const forged = r.ticks.filter((t) => t.attack.active);
    expect(forged.length).toBeGreaterThan(200);
    expect(forged.every((t) => t.verdict.l0.ok && t.verdict.l0.crc_ok)).toBe(true);
  });

  it("catches bias almost instantly with an L1 physics violation", () => {
    const r = run({ attack: "bias", seconds: 80 });
    expect(firstAlertT(r)).toBeLessThan(61);
    expect(r.alerts[0]!.verdict.l1_violations.map((v) => v.rule)).toContain("speed_vs_components");
    expect(r.alerts[0]!.spoken).toMatch(/^Flight, GNC\./);
  });

  it("catches drift through L2 after a short delay, without L1", () => {
    const r = run({ attack: "drift", seconds: 100 });
    const t = firstAlertT(r);
    expect(t).toBeGreaterThan(62);
    expect(t).toBeLessThan(85);
    expect(r.alerts[0]!.verdict.l1_violations).toHaveLength(0);
  });

  it("catches replay", () => {
    const r = run({ attack: "replay", seconds: 70 });
    expect(firstAlertT(r)).toBeLessThan(62);
  });

  it("catches the coordinated attack despite consistent kinematics", () => {
    const r = run({ attack: "coordinated", seconds: 110 });
    expect(firstAlertT(r)).toBeLessThan(100);
    expect(r.alerts[0]!.verdict.l1_violations).toHaveLength(0);
  });

  it("lets PPO slip past the base detector while the track diverges", () => {
    const r = run({ attack: "ppo", seconds: 150 });
    expect(r.alerts).toHaveLength(0);
    const last = r.ticks[r.ticks.length - 1]!;
    const gap = Math.abs(last.reported.channels.altitude! - last.truth.channels.altitude!);
    expect(gap).toBeGreaterThan(2000);
  });

  it("catches PPO once the hardened detector is in", () => {
    const r = run({ attack: "ppo", detector: "hardened", seconds: 120 });
    expect(firstAlertT(r)).toBeLessThan(95);
    expect(r.alerts[0]!.verdict.l3_prob).toBeGreaterThan(0.5);
  });

  it("hot-swaps the detector mid-run", () => {
    const sim = new MockSim(1);
    sim.control({ action: "start", scenario: "falcon9:ses-9", attack: "ppo", attack_params: { start_t: 10, intensity: 0.5 } });
    for (let i = 0; i < 400; i++) sim.advance(0.05);
    const swapped = sim.setDetector({ detector: "hardened" });
    expect(swapped[0]).toMatchObject({ type: "status", status: { detector: "hardened" } });
    const msgs: ServerMessage[] = [];
    for (let i = 0; i < 100; i++) msgs.push(...sim.advance(0.05));
    expect(msgs.some((m) => m.type === "alert")).toBe(true);
  });

  it("runs quad and spacecraft scenarios with their schemas", () => {
    const quad = run({ scenario: "quad:figure8", seconds: 20, startT: 1000 });
    expect(quad.ticks[0]!.truth.schema_id).toBe(2);
    expect(Object.keys(quad.ticks[0]!.truth.channels)).toHaveLength(18);
    expect(quad.alerts).toHaveLength(0);
    const sc = run({ scenario: "smap:P-1", seconds: 5 });
    expect(sc.ticks[0]!.truth.schema_id).toBe(3);
  });

  it("reset rewinds to t=0 and announces the scenario", () => {
    const sim = new MockSim(3);
    sim.control({ action: "start" });
    for (let i = 0; i < 40; i++) sim.advance(0.05);
    expect(sim.getStatus().t).toBeGreaterThan(1.5);
    const msgs = sim.control({ action: "reset" });
    expect(msgs[0]?.type).toBe("scenario");
    expect(sim.getStatus()).toMatchObject({ t: 0, running: false, frames: 0 });
  });
});
