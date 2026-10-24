import { describe, expect, it } from "vitest";
import { Extent, linear, niceDomain, niceStep, niceTicks } from "@/lib/canvas/scale";

describe("scales", () => {
  it("picks 1-2-5 steps", () => {
    expect(niceStep(0.7)).toBe(1);
    expect(niceStep(1.3)).toBe(2);
    expect(niceStep(3)).toBe(5);
    expect(niceStep(7)).toBe(10);
    expect(niceStep(240)).toBe(500);
  });
  it("produces round ticks inside the range", () => {
    expect(niceTicks(0, 100, 5)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(niceTicks(-0.3, 0.3, 3)).toEqual([-0.2, 0, 0.2]);
    expect(niceTicks(5, 5)).toEqual([5]);
  });
  it("expands domains to nice bounds with a minimum span", () => {
    expect(niceDomain(3, 97)).toEqual([0, 100]);
    const [lo, hi] = niceDomain(10, 10, 2);
    expect(lo).toBeLessThanOrEqual(9);
    expect(hi).toBeGreaterThanOrEqual(11);
  });
  it("maps linearly and tracks extents ignoring NaN", () => {
    const f = linear([0, 10], [100, 0]);
    expect(f(5)).toBe(50);
    const e = new Extent();
    expect(e.empty).toBe(true);
    [3, Number.NaN, -1, 8].forEach((v) => e.add(v));
    expect([e.min, e.max]).toEqual([-1, 8]);
  });
});
