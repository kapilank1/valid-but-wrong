import { describe, expect, it } from "vitest";
import { parseAnomalyRanges, parseEvents, resolveSchema } from "@/lib/scenario";

describe("resolveSchema", () => {
  it("prefers the frame schema id", () => {
    expect(resolveSchema({ id: "falcon9:x", schema: "falcon9" }, 2)).toBe("quad");
  });
  it("falls back to the scenario schema field, then the id prefix", () => {
    expect(resolveSchema({ id: "x", schema: "spacecraft" })).toBe("spacecraft");
    expect(resolveSchema({ id: "x", schema: 1 })).toBe("falcon9");
    expect(resolveSchema({ id: "msl:M-1", schema: "?" })).toBe("spacecraft");
    expect(resolveSchema({ id: "sitl:udp:14550", schema: "?" })).toBe("quad");
    expect(resolveSchema(null)).toBeNull();
  });
});

describe("parseEvents", () => {
  it("returns numeric events sorted by time", () => {
    expect(parseEvents({ events: { meco: 161, maxq: 81, anomalies: [[1, 2]] } })).toEqual([
      { name: "maxq", t: 81 },
      { name: "meco", t: 161 },
    ]);
    expect(parseEvents({ events: null })).toEqual([]);
  });
});

describe("parseAnomalyRanges", () => {
  it("accepts pairs, objects and named pair events", () => {
    expect(parseAnomalyRanges({ events: { anomalies: [[30, 10]] } })).toEqual([{ start: 10, end: 30, label: undefined }]);
    expect(parseAnomalyRanges({ events: null, anomalies: [{ start: 5, end: 6, class: "point" }] })).toEqual([
      { start: 5, end: 6, label: "point" },
    ]);
    expect(parseAnomalyRanges({ events: { anomaly_1: [100, 120], maxq: 3 } })).toEqual([
      { start: 100, end: 120, label: "anomaly_1" },
    ]);
  });
});
