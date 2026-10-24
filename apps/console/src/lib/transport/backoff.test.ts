import { describe, expect, it } from "vitest";
import { backoffDelay } from "@/lib/transport/backoff";

describe("backoffDelay", () => {
  const noJitter = { baseMs: 500, maxMs: 8000, jitter: 0 };
  it("doubles per attempt up to the cap", () => {
    expect([0, 1, 2, 3, 4, 5, 10].map((a) => backoffDelay(a, noJitter))).toEqual([500, 1000, 2000, 4000, 8000, 8000, 8000]);
  });
  it("applies bounded jitter", () => {
    const opts = { baseMs: 1000, maxMs: 8000, jitter: 0.2 };
    expect(backoffDelay(0, opts, () => 0)).toBe(800);
    expect(backoffDelay(0, opts, () => 1)).toBe(1200);
    expect(backoffDelay(0, opts, () => 0.5)).toBe(1000);
  });
});
