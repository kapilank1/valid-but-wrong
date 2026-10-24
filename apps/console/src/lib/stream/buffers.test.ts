import { describe, expect, it } from "vitest";
import { TelemetryBuffers } from "@/lib/stream/buffers";
import type { TickMessage } from "@/lib/types";
import { makeTick } from "@/lib/stream/testUtils";

describe("TelemetryBuffers", () => {
  it("stores truth and reported per channel, index-aligned with t", () => {
    const b = new TelemetryBuffers(10);
    b.push(makeTick(0.1, { altitude: 10 }, { altitude: 12 }));
    b.push(makeTick(0.2, { altitude: 20 }, { altitude: 25 }, { score: 0.8, alert: true, active: true }));
    expect(b.length).toBe(2);
    expect(b.t.toArray()).toEqual([0.1, 0.2]);
    expect(b.truth.get("altitude")!.toArray()).toEqual([10, 20]);
    expect(b.reported.get("altitude")!.toArray()).toEqual([12, 25]);
    expect(b.score.toArray()).toEqual([0, 0.8]);
    expect(b.alert.toArray()).toEqual([0, 1]);
    expect(b.attackActive.toArray()).toEqual([0, 1]);
  });

  it("keeps late and missing channels aligned with NaN", () => {
    const b = new TelemetryBuffers(10);
    b.push(makeTick(1, { a: 1 }, { a: 1 }));
    b.push(makeTick(2, { a: 2, b: 5 }, { a: 2, b: 5 }));
    b.push(makeTick(3, { b: 6 }, { b: 6 }));
    expect(b.truth.get("a")!.toArray()).toEqual([1, 2, Number.NaN]);
    expect(b.truth.get("b")!.toArray()).toEqual([Number.NaN, 5, 6]);
  });

  it("bumps version on every mutation and finds time indices", () => {
    const b = new TelemetryBuffers(10);
    const v0 = b.version;
    for (let i = 0; i < 5; i++) b.push(makeTick(i, { x: i }, { x: i }) as TickMessage);
    expect(b.version).toBe(v0 + 5);
    expect(b.indexAtOrAfter(2.5)).toBe(3);
    b.clear();
    expect(b.length).toBe(0);
    expect(b.truth.size).toBe(0);
    expect(b.version).toBe(v0 + 6);
  });
});
