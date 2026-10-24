import type { TruthModel } from "@/lib/mock/vehicles";
import type { AttackInfo, Catalog, DetectorInfo } from "@/lib/types";

const ALL_SCHEMAS = ["falcon9", "quad", "spacecraft"];

const ATTACKS: Omit<AttackInfo, "params">[] = [
  { name: "none", label: "None", description: "Identity: frames pass through untouched", schemas: ALL_SCHEMAS },
  { name: "bias", label: "Bias", description: "Constant offset on one channel", schemas: ALL_SCHEMAS },
  { name: "drift", label: "Drift", description: "Slow ramp that stays under per-frame rate limits", schemas: ALL_SCHEMAS },
  { name: "replay", label: "Replay", description: "Re-send old in-sequence frames with fresh seq numbers", schemas: ALL_SCHEMAS },
  {
    name: "coordinated",
    label: "Coordinated",
    description: "Moves position and velocity together so kinematic rules agree",
    schemas: ALL_SCHEMAS,
  },
  {
    name: "ppo",
    label: "PPO agent",
    description: "Learned policy that maximizes operator error minus detection",
    schemas: ALL_SCHEMAS,
  },
];

const DETECTORS: DetectorInfo[] = [
  { name: "base", label: "Base", description: "L1 physics + L2 GRU residuals" },
  { name: "hardened", label: "Hardened", description: "Base + L3 discriminator retrained on PPO frames" },
];

export function mockCatalog(models: TruthModel[]): Catalog {
  return {
    scenarios: models.map((m) => m.info),
    attacks: ATTACKS.map((a) => ({ ...a, params: { start_t: 60, intensity: 0.5 } })),
    detectors: DETECTORS,
  };
}

/**
 * Placeholder bench numbers for mock mode only (the card is labeled MOCK). The baseline row uses
 * the Telemanom results reported in Hundman et al., KDD 2018.
 */
export const MOCK_BENCH = {
  available: true,
  mock: true,
  metric: "event-level precision / recall / F1",
  datasets: {
    SMAP: { precision: 0.871, recall: 0.862, f1: 0.866, channels: 54 },
    MSL: { precision: 0.903, recall: 0.741, f1: 0.814, channels: 27 },
  },
  baseline: {
    name: "Telemanom (Hundman et al. 2018)",
    SMAP: { precision: 0.855, recall: 0.855, f1: 0.855 },
    MSL: { precision: 0.926, recall: 0.694, f1: 0.793 },
  },
  latency_ms: { p50: 2.1, p99: 6.8 },
};
