import { describe, expect, it } from "vitest";
import { attackAvailability, detectorAvailability, detectorThreshold } from "@/lib/catalog";
import type { Catalog } from "@/lib/types";

const catalog: Catalog = {
  scenarios: [],
  attacks: [
    { name: "bias", label: "Bias", description: "", schemas: ["falcon9", "quad"], params: { start_t: 60, intensity: 0.5 } },
    { name: "ppo", label: "PPO", description: "", schemas: [], params: {}, available: false },
  ],
  detectors: [
    { name: "base", label: "Base", description: "" },
    { name: "hardened", label: "Hardened", description: "", available: false },
  ],
};

describe("catalog availability", () => {
  it("checks presence, explicit availability and schema support", () => {
    expect(attackAvailability(catalog, "bias", "falcon9").available).toBe(true);
    expect(attackAvailability(catalog, "bias", "spacecraft")).toMatchObject({ available: false });
    expect(attackAvailability(catalog, "ppo", "falcon9").available).toBe(false);
    expect(attackAvailability(catalog, "replay", "falcon9").reason).toContain("catalog");
    expect(attackAvailability(null, "bias", null).available).toBe(false);
  });

  it("checks detectors and reads an optional threshold", () => {
    expect(detectorAvailability(catalog, "base").available).toBe(true);
    expect(detectorAvailability(catalog, "hardened").available).toBe(false);
    expect(detectorThreshold(catalog, "base")).toBe(0.5);
    const withThreshold: Catalog = { ...catalog, detectors: [{ name: "base", label: "", description: "", threshold: 0.7 }] };
    expect(detectorThreshold(withThreshold, "base")).toBe(0.7);
  });
});
