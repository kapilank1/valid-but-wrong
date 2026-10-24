import { describe, expect, it } from "vitest";
import { initialControls, resetRequest, startRequest } from "@/lib/controls";
import { MockSim } from "@/lib/mock/sim";

describe("run controls", () => {
  const sim = new MockSim(1);

  it("initializes from the server status", () => {
    const c = initialControls(sim.getStatus(), sim.catalog);
    expect(c).toEqual({ scenario: "falcon9:ses-9", attack: "none", intensity: 0.5, startT: 60, detector: "base", speed: 1 });
  });

  it("builds contract-shaped start and reset bodies", () => {
    const c = { scenario: "quad:figure8", attack: "drift", intensity: 0.333, startT: -4, detector: "hardened", speed: 2 };
    expect(startRequest(c)).toEqual({
      action: "start",
      scenario: "quad:figure8",
      attack: "drift",
      attack_params: { intensity: 0.33, start_t: 0 },
      detector: "hardened",
      speed: 2,
    });
    expect(resetRequest(c).action).toBe("reset");
  });
});
