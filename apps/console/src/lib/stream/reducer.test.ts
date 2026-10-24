import { describe, expect, it } from "vitest";
import { MockSim } from "@/lib/mock/sim";
import {
  ALERT_HOLD_S,
  crcPassRate,
  initialStreamState,
  reduceStream,
  selectAlertBanner,
  type StreamAction,
  type StreamState,
} from "@/lib/stream/reducer";
import { makeTick } from "@/lib/stream/testUtils";

const reduceAll = (actions: StreamAction[], start: StreamState = initialStreamState) => actions.reduce(reduceStream, start);
const hello = () => new MockSim(1).hello()[0]!;

describe("reduceStream", () => {
  it("stores catalog and status from hello and resolves the running scenario", () => {
    const s = reduceStream(initialStreamState, hello());
    expect(s.catalog?.scenarios.length).toBeGreaterThan(0);
    expect(s.status?.scenario).toBe("falcon9:ses-9");
    expect(s.scenario?.id).toBe("falcon9:ses-9");
  });

  it("counts L0 results and forged frames", () => {
    const s = reduceAll([
      makeTick(1, { a: 1 }, { a: 1 }),
      makeTick(2, { a: 2 }, { a: 9 }, { active: true }),
      makeTick(3, { a: 3 }, { a: 9 }, { active: true, crcOk: false }),
    ]);
    expect(s.link).toMatchObject({ frames: 3, crc: 2, sync: 3, allOk: 2, forged: 2, forgedPassing: 1 });
    expect(crcPassRate(s.link)).toBeCloseTo(2 / 3);
    expect(crcPassRate(initialStreamState.link)).toBe(1);
  });

  it("logs attack and alert edges newest-first", () => {
    const s = reduceAll([
      makeTick(1, {}, {}),
      makeTick(2, {}, {}, { active: true, attack: "drift" }),
      makeTick(3, {}, {}, { active: true, attack: "drift", alert: true, score: 0.9 }),
      { type: "alert", verdict: makeTick(3, {}, {}, { alert: true, score: 0.9 }).verdict, spoken: "Flight, GNC." },
      makeTick(4, {}, {}, { active: false }),
    ]);
    expect(s.log.map((e) => e.kind)).toEqual(["clear", "attack", "alert", "attack"]);
    expect(s.log[3]!.text).toContain("DRIFT");
    expect(s.alertCount).toBe(1);
    expect(s.lastAlert?.spoken).toBe("Flight, GNC.");
  });

  it("resets run state on a new scenario but keeps the log", () => {
    const sim = new MockSim(1);
    const scenario = sim.hello()[1]!;
    const s = reduceAll([makeTick(1, {}, {}, { active: true }), scenario]);
    expect(s.latest).toBeNull();
    expect(s.link.frames).toBe(0);
    expect(s.log[0]!.kind).toBe("scenario");
    expect(s.log.length).toBe(2);
  });

  it("treats mission time going backwards as a new run", () => {
    const s = reduceAll([makeTick(50, {}, {}), makeTick(51, {}, {}), makeTick(0.1, {}, {})]);
    expect(s.link.frames).toBe(1);
  });

  it("logs detector hot swaps from status", () => {
    const h = hello();
    if (h.type !== "hello") throw new Error("expected hello");
    const s = reduceAll([h, { type: "status", status: { ...h.status, detector: "hardened" } }]);
    expect(s.log[0]).toMatchObject({ kind: "detector" });
    expect(s.log[0]!.text).toContain("HARDENED");
  });

  it("caps the log", () => {
    const actions: StreamAction[] = Array.from({ length: 250 }, (_, i) => ({ type: "local:log", kind: "info", text: `m${i}` }));
    const s = reduceAll(actions);
    expect(s.log.length).toBe(200);
    expect(s.log[0]!.text).toBe("m249");
  });
});

describe("selectAlertBanner", () => {
  it("is active while alerting and holds briefly after it clears", () => {
    const alerting = reduceAll([
      makeTick(10, {}, {}, { alert: true, score: 0.9 }),
      { type: "alert", verdict: makeTick(10, {}, {}, { alert: true }).verdict, spoken: "x" },
    ]);
    expect(selectAlertBanner(alerting)).toMatchObject({ active: true, live: true });
    const held = reduceStream(alerting, makeTick(10 + ALERT_HOLD_S - 1, {}, {}));
    expect(selectAlertBanner(held)).toMatchObject({ active: true, live: false });
    const cleared = reduceStream(held, makeTick(10 + ALERT_HOLD_S + 1, {}, {}));
    expect(selectAlertBanner(cleared).active).toBe(false);
  });

  it("falls back to the verdict when the tick beats its alert message", () => {
    const s = reduceStream(initialStreamState, makeTick(5, {}, {}, { alert: true, score: 0.7 }));
    const view = selectAlertBanner(s);
    expect(view.active).toBe(true);
    expect(view.alert?.explanation).toBe("explained");
  });
});
