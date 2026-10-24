import type { TruthModel } from "@/lib/mock/vehicles";
import type { SchemaName } from "@/lib/types";

/**
 * In-browser stand-ins for vbw/attacker. Each one rewrites the true channels into the lie the
 * attacker wants the ground to see; the link layer would re-encode it with a valid CRC.
 */

export const MOCK_ATTACKS = ["none", "bias", "drift", "replay", "coordinated", "ppo"] as const;
export type MockAttackName = (typeof MOCK_ATTACKS)[number];

export function isMockAttack(name: string): name is MockAttackName {
  return (MOCK_ATTACKS as readonly string[]).includes(name);
}

export interface AttackContext {
  model: TruthModel;
  t: number;
  startT: number;
  intensity: number; // 0..1
}

export interface AttackResult {
  channels: Record<string, number>;
  active: boolean;
}

type Channels = Record<string, number>;
type Offsets = Record<string, number>;

/** Position/velocity pairs the attacker moves together so kinematics stay self-consistent. */
const PAIRS: Record<SchemaName, { pos: string; vel: string; gain: number }[]> = {
  falcon9: [
    { pos: "altitude", vel: "vz", gain: 1 },
    { pos: "downrange", vel: "vx", gain: 1.5 },
  ],
  quad: [
    { pos: "pn", vel: "vn", gain: 1 },
    { pos: "pe", vel: "ve", gain: 0.7 },
  ],
  spacecraft: [],
};

export function applyMockAttack(name: string, ctx: AttackContext, truth: Channels): AttackResult {
  const tau = ctx.t - ctx.startT;
  if (name === "none" || !isMockAttack(name) || tau < 0) return { channels: truth, active: false };
  const schema = ctx.model.schema.name;
  const I = Math.min(1, Math.max(0, ctx.intensity));

  if (name === "replay") {
    // Old, in-sequence frames: each is internally consistent, but the vehicle is "back in time".
    const lag = schema === "quad" ? 10 + 20 * I : 20 + 40 * I;
    return { channels: ctx.model.sample(Math.max(0, ctx.t - lag)), active: true };
  }

  const offsetsAt = (tt: number) => attackOffsets(name, schema, I, tt);
  const out = withOffsets(truth, offsetsAt(tau));
  // Bias and drift are naive: they touch one channel and leave the derived ones alone (L1 bait).
  // The smarter attacks also patch derived channels so cross-channel rules stay quiet.
  if (name === "coordinated" || name === "ppo") patchDerived(schema, ctx, truth, out, offsetsAt);
  return { channels: out, active: true };
}

function attackOffsets(name: MockAttackName, schema: SchemaName, I: number, tau: number): Offsets {
  const o: Offsets = {};
  switch (name) {
    case "bias":
      if (schema === "falcon9") o.speed = 150 * I;
      if (schema === "quad") Object.assign(o, { alt: 10 * I, pn: 20 * I });
      if (schema === "spacecraft") o.value = 0.4 * I;
      break;
    case "drift": // slow enough to stay under the L1 rate rules
      if (schema === "falcon9") o.altitude = 80 * I * tau;
      if (schema === "quad") Object.assign(o, { pn: 1.2 * I * tau, pe: 0.8 * I * tau });
      if (schema === "spacecraft") o.value = 0.02 * I * tau;
      break;
    case "coordinated": {
      // A constant phantom acceleration injected into position and velocity together.
      const accel = { falcon9: 3, quad: 0.15, spacecraft: 0 }[schema] * I;
      for (const { pos, vel, gain } of PAIRS[schema]) {
        o[vel] = gain * accel * tau;
        o[pos] = (gain * accel * tau * tau) / 2;
      }
      if (schema === "spacecraft") o.value = 0.0004 * I * tau * tau;
      break;
    }
    case "ppo": {
      // Stand-in for the learned policy: a smooth, kinematically consistent walk-off (cycloid) that
      // keeps growing the operator's error while each step stays small. `perPeriod` = error per 90 s.
      const perPeriod = { falcon9: 12000, quad: 40, spacecraft: 0 }[schema] * I;
      const omega = (2 * Math.PI) / 90;
      PAIRS[schema].forEach(({ pos, vel, gain }, k) => {
        const w = omega * (1 - 0.3 * k);
        o[pos] = (gain * perPeriod * (w * tau - Math.sin(w * tau))) / (2 * Math.PI);
        o[vel] = (gain * perPeriod * w * (1 - Math.cos(w * tau))) / (2 * Math.PI);
      });
      if (schema === "spacecraft") o.value = 0.3 * I * (1 - Math.cos(omega * tau));
      break;
    }
    case "none":
    case "replay":
      break;
  }
  return o;
}

function withOffsets(c: Channels, offsets: Offsets): Channels {
  const out = { ...c };
  for (const [k, d] of Object.entries(offsets)) {
    const v = out[k];
    if (typeof v === "number") out[k] = v + d;
  }
  return out;
}

/** Keep speed/groundspeed (and Falcon 9 accel) consistent with the forged velocity components. */
function patchDerived(
  schema: SchemaName,
  ctx: AttackContext,
  truth: Channels,
  out: Channels,
  offsetsAt: (tau: number) => Offsets,
): void {
  if (schema === "quad") {
    out.groundspeed = (truth.groundspeed ?? 0) + Math.hypot(out.vn ?? 0, out.ve ?? 0) - Math.hypot(truth.vn ?? 0, truth.ve ?? 0);
  }
  if (schema === "falcon9") {
    const speedOffset = (tt: number, tr: Channels) => {
      const o = offsetsAt(tt - ctx.startT);
      const vx = tr.vx ?? 0;
      const vz = tr.vz ?? 0;
      return Math.hypot(vx + (o.vx ?? 0), vz + (o.vz ?? 0)) - Math.hypot(vx, vz);
    };
    const h = 0.5;
    const now = speedOffset(ctx.t, truth);
    const before = speedOffset(ctx.t - h, ctx.model.sample(ctx.t - h));
    out.speed = (truth.speed ?? 0) + now;
    out.accel = (truth.accel ?? 0) + (now - before) / h;
  }
}
