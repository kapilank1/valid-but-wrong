import { describe, expect, it } from "vitest";
import { formatMissionClock, formatStamp, formatValue } from "@/lib/format";

describe("format", () => {
  it("formats the mission clock", () => {
    expect(formatMissionClock(133.45)).toBe("T+ 00:02:13.4");
    expect(formatMissionClock(3725)).toBe("T+ 01:02:05.0");
    expect(formatMissionClock(-5)).toBe("T- 00:00:05.0");
    expect(formatMissionClock(null)).toBe("T+ --:--:--.-");
  });
  it("formats log stamps and values", () => {
    expect(formatStamp(61.25)).toBe("T+01:01.2");
    expect(formatValue(1234.5)).toBe("1,235");
    expect(formatValue(12.345)).toBe("12.35");
    expect(formatValue(Number.NaN)).toBe("—");
  });
});
