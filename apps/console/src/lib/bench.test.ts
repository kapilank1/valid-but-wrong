import { describe, expect, it } from "vitest";
import { normalizeBench, readPrf } from "@/lib/bench";
import { MOCK_BENCH } from "@/lib/mock/catalog";

describe("readPrf", () => {
  it("reads fractions, percentages and derives F1", () => {
    expect(readPrf({ precision: 0.5, recall: 0.5, f1: 0.5 })).toEqual({ precision: 0.5, recall: 0.5, f1: 0.5 });
    expect(readPrf({ P: 80, R: 60 })!.f1).toBeCloseTo((2 * 0.8 * 0.6) / 1.4);
    expect(readPrf({ precision: 1 })).toBeNull();
  });
});

describe("normalizeBench", () => {
  it("returns null when unavailable", () => {
    expect(normalizeBench({ available: false })).toBeNull();
    expect(normalizeBench(null)).toBeNull();
    expect(normalizeBench({ foo: 1 })).toBeNull();
  });

  it("reads the datasets + baseline layout", () => {
    const v = normalizeBench(MOCK_BENCH)!;
    expect(v.rows.map((r) => r.dataset)).toEqual(["SMAP", "MSL"]);
    expect(v.rows[0]!.baseline?.precision).toBeCloseTo(0.855);
    expect(v.baselineName).toContain("Telemanom");
    expect(v.latencyMs).toBeCloseTo(2.1);
    expect(v.mock).toBe(true);
  });

  it("reads per-dataset ours/baseline objects", () => {
    const v = normalizeBench({ smap: { ours: { p: 0.9, r: 0.8, f1: 0.85 }, baseline: { p: 0.85, r: 0.85, f1: 0.85 } }, baseline_name: "Telemanom" })!;
    expect(v.rows).toHaveLength(1);
    expect(v.rows[0]).toMatchObject({ dataset: "SMAP", ours: { precision: 0.9 }, baseline: { recall: 0.85 } });
    expect(v.baselineName).toBe("Telemanom");
  });

  it("reads a results list", () => {
    const v = normalizeBench({ results: [{ dataset: "MSL", precision: 92, recall: 70, f1: 79.5 }] })!;
    expect(v.rows[0]).toMatchObject({ dataset: "MSL", ours: { precision: 0.92, f1: 0.795 }, baseline: null });
    expect(v.baselineName).toBeNull();
  });
});
