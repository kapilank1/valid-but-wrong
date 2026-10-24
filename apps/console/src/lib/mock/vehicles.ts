import { FALCON9_PROFILE } from "@/lib/mock/falcon9Profile";
import { FALCON9, QUAD, SPACECRAFT, type ChannelSchema, type ScenarioInfo } from "@/lib/types";

/** Ground truth for one mock scenario: a deterministic function of mission time. */
export interface TruthModel {
  info: ScenarioInfo;
  schema: ChannelSchema;
  /** Default `start_t` for attacks on this scenario (mission seconds). */
  defaultAttackStart: number;
  /** Per-channel 1σ scale used by the mock L2 residuals and attack magnitudes. */
  scales: Record<string, number>;
  sample(t: number): Record<string, number>;
}

// --------------------------------------------------------------------------- Falcon 9 (real data)

function falcon9Model(): TruthModel {
  const { rows, channels, dt, events, slug, missionName } = FALCON9_PROFILE;
  const duration = (rows.length - 1) * dt;
  return {
    info: {
      id: `falcon9:${slug}`,
      label: `Falcon 9 · ${missionName} ascent`,
      schema: "falcon9",
      vehicle_id: `F9-${slug.toUpperCase()}`,
      rate_hz: 10,
      duration_s: duration,
      events,
      description: "Real webcast telemetry replay (mock stream)",
    },
    schema: FALCON9,
    defaultAttackStart: 60,
    scales: { speed: 15, altitude: 120, vz: 12, vx: 15, downrange: 200, accel: 1.5, angle: 0.8, q: 600 },
    sample(t) {
      const x = Math.min(Math.max(t / dt, 0), rows.length - 1);
      const i = Math.min(Math.floor(x), rows.length - 2);
      const f = x - i;
      const a = rows[i] ?? [];
      const b = rows[i + 1] ?? a;
      const out: Record<string, number> = {};
      channels.forEach((name, k) => {
        const va = a[k] ?? 0;
        out[name] = va + ((b[k] ?? va) - va) * f;
      });
      return out;
    },
  };
}

// --------------------------------------------------------------------------- quad (synthetic)

const G = 9.81;

function quadModel(): TruthModel {
  const w = (2 * Math.PI) / 80; // one figure-8 lap every 80 s
  const climb = (t: number) => 25 * (1 - Math.exp(-t / 5));
  return {
    info: {
      id: "quad:figure8",
      label: "Quad · figure-8 survey (sim)",
      schema: "quad",
      vehicle_id: "QUAD-01",
      rate_hz: 20,
      duration_s: 600,
      events: { takeoff: 0, survey_start: 10 },
      description: "Simulated multirotor, local NED frame (mock stream)",
    },
    schema: QUAD,
    defaultAttackStart: 25,
    scales: {
      pn: 1.5, pe: 1.5, alt: 0.8, vn: 0.4, ve: 0.4, vu: 0.3,
      roll: 0.05, pitch: 0.05, yaw: 0.1, p: 0.1, q: 0.1, r: 0.1,
      ax: 0.5, ay: 0.5, az: 0.5, baro_alt: 0.8, groundspeed: 0.4, battery_v: 0.05,
    },
    sample(t) {
      const ramp = Math.min(1, Math.max(0, (t - 5) / 10)); // ease into the pattern after climb-out
      const dramp = t > 5 && t < 15 ? 0.1 : 0;
      const pn = ramp * 60 * Math.sin(w * t);
      const pe = ramp * 40 * Math.sin(2 * w * t);
      // Product rule so velocity stays exactly consistent with position during the ramp.
      const vn = dramp * 60 * Math.sin(w * t) + ramp * 60 * w * Math.cos(w * t);
      const ve = dramp * 40 * Math.sin(2 * w * t) + ramp * 80 * w * Math.cos(2 * w * t);
      const an = -ramp * 60 * w * w * Math.sin(w * t);
      const ae = -ramp * 160 * w * w * Math.sin(2 * w * t);
      const alt = climb(t);
      const vu = (25 / 5) * Math.exp(-t / 5);
      const yaw = Math.atan2(ve, vn || 1e-6);
      return {
        pn, pe, alt, vn, ve, vu,
        roll: ae / G, pitch: -an / G, yaw,
        p: (ramp * -160 * w * w * w * Math.cos(2 * w * t)) / G,
        q: (ramp * 60 * w * w * w * Math.cos(w * t)) / G,
        r: 0.4 * w * Math.cos(w * t),
        ax: an * 0.2, ay: ae * 0.2, az: -G,
        baro_alt: alt + 0.3,
        groundspeed: Math.hypot(vn, ve),
        battery_v: 16.8 - 0.003 * t,
      };
    },
  };
}

// --------------------------------------------------------------------------- spacecraft (synthetic)

export const MOCK_SPACECRAFT_ANOMALIES: [number, number][] = [
  [150, 175],
  [380, 395],
];

function spacecraftModel(): TruthModel {
  return {
    info: {
      id: "smap:P-1",
      label: "SMAP P-1 · power channel (synthetic)",
      schema: "spacecraft",
      vehicle_id: "SMAP",
      rate_hz: 10,
      duration_s: 600,
      events: { anomalies: MOCK_SPACECRAFT_ANOMALIES },
      description: "Synthetic stand-in for a SMAP channel with labeled anomalies (mock stream)",
    },
    schema: SPACECRAFT,
    defaultAttackStart: 220,
    scales: { value: 0.05 },
    sample(t) {
      let value = 0.35 * Math.sin((2 * Math.PI * t) / 45) + 0.12 * Math.sin((2 * Math.PI * t) / 9.7);
      // The labeled anomalies are real-world faults in the source data, not attacks.
      const [a0, a1] = MOCK_SPACECRAFT_ANOMALIES;
      if (a0 && t >= a0[0] && t <= a0[1]) value += 0.45;
      if (a1 && t >= a1[0] && t <= a1[1]) value += 0.3 * Math.sin(2 * Math.PI * 1.5 * t);
      return { value };
    },
  };
}

export function mockTruthModels(): TruthModel[] {
  return [falcon9Model(), quadModel(), spacecraftModel()];
}
