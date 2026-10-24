import { describe, expect, it } from "vitest";
import { EMPTY_LINK } from "@/lib/stream/reducer";
import { makeTick } from "@/lib/stream/testUtils";
import { altitudeLieKm, linkValidity, truthGap } from "@/lib/truthGap";

describe("truthGap", () => {
  it("formats a falcon9 altitude lie in km with severity", () => {
    const gap = truthGap(makeTick(200, { altitude: 120_000 }, { altitude: 123_720 }), "falcon9");
    expect(gap).toMatchObject({ text: "3.72", unit: "km", channel: "ALTITUDE", level: "large" });
  });

  it("uses metres below a kilometre and calls sensor-sized gaps none", () => {
    expect(truthGap(makeTick(1, { altitude: 1000 }, { altitude: 1300 }), "falcon9")).toMatchObject({ text: "300", unit: "m", level: "small" });
    expect(truthGap(makeTick(1, { altitude: 1000 }, { altitude: 1000 }), "falcon9")?.level).toBe("none");
  });

  it("measures the quad position gap in 3D and returns null without data", () => {
    expect(truthGap(makeTick(1, { pn: 0, pe: 0, alt: 10 }, { pn: 3, pe: 4, alt: 10 }), "quad")).toMatchObject({ text: "5.0", level: "large" });
    expect(truthGap(null, "falcon9")).toBeNull();
    expect(truthGap(makeTick(1, {}, {}), "falcon9")).toBeNull();
  });

  it("reports the signed altitude lie in km", () => {
    expect(altitudeLieKm(makeTick(1, { altitude: 5000 }, { altitude: 3500 }))).toBeCloseTo(-1.5);
    expect(altitudeLieKm(null)).toBeNull();
  });
});

describe("linkValidity", () => {
  it("is 100% when every frame passed, with forged counts alongside", () => {
    const v = linkValidity({ ...EMPTY_LINK, frames: 50, allOk: 50, forged: 20, forgedPassing: 20 });
    expect(v).toMatchObject({ fraction: 1, text: "100%", forged: 20, forgedPassing: 20 });
  });

  it("shows a dash before the first frame and one decimal otherwise", () => {
    expect(linkValidity(EMPTY_LINK).text).toBe("—");
    expect(linkValidity({ ...EMPTY_LINK, frames: 1000, allOk: 997 }).text).toBe("99.7%");
  });
});
