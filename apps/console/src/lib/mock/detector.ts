import type { Rng } from "@/lib/mock/rng";
import type { TruthModel } from "@/lib/mock/vehicles";
import type { Frame, SchemaName, Verdict, Violation } from "@/lib/types";

/**
 * In-browser stand-in for vbw/detector so the console can be demoed without the backend.
 * L1 runs honest consistency rules on the reported stream only. L2/L3 are faked from the
 * truth-vs-reported gap (the mock can see truth; the real detector can't) and tuned to reproduce
 * the demo narrative: PPO slips past "base" and is caught by "hardened".
 */

type Channels = Record<string, number>;

interface Window {
  now: Channels;
  then: Channels;
  dt: number;
  mean(channel: string): number;
}

interface Rule {
  rule: string;
  channels: string[];
  windowS: number;
  threshold: number;
  measure(w: Window): number;
  message(value: number, w: Window): string;
}

const fmt = (v: number, d = 0) => v.toFixed(d);
const get = (c: Channels, k: string) => c[k] ?? 0;
const rate = (w: Window, k: string) => (get(w.now, k) - get(w.then, k)) / w.dt;

const RULES: Record<SchemaName, Rule[]> = {
  falcon9: [
    {
      rule: "speed_vs_components",
      channels: ["speed", "vx", "vz"],
      windowS: 0,
      threshold: 30,
      measure: (w) => Math.abs(get(w.now, "speed") - Math.hypot(get(w.now, "vx"), get(w.now, "vz"))),
      message: (v, w) =>
        `Reported speed ${fmt(get(w.now, "speed"))} m/s doesn't match its velocity components (off by ${fmt(v)} m/s)`,
    },
    {
      // 5 s window: the webcast altitude has coarse steps that make shorter windows noisy.
      rule: "vz_vs_dalt",
      channels: ["vz", "altitude"],
      windowS: 5,
      threshold: 120,
      measure: (w) => Math.abs(w.mean("vz") - rate(w, "altitude")),
      message: (v, w) =>
        `Altitude changing ${fmt(rate(w, "altitude"))} m/s but vertical velocity says ${fmt(w.mean("vz"))} m/s`,
    },
    {
      rule: "vx_vs_ddownrange",
      channels: ["vx", "downrange"],
      windowS: 1,
      threshold: 8,
      measure: (w) => Math.abs(w.mean("vx") - rate(w, "downrange")),
      message: (v) => `Downrange rate disagrees with horizontal velocity by ${fmt(v)} m/s`,
    },
    {
      rule: "accel_vs_dspeed",
      channels: ["accel", "speed"],
      windowS: 1,
      threshold: 25,
      measure: (w) => Math.abs(w.mean("accel") - rate(w, "speed")),
      message: (v) => `Reported acceleration disagrees with speed change by ${fmt(v, 1)} m/s²`,
    },
  ],
  quad: [
    {
      rule: "baro_vs_gps_alt",
      channels: ["alt", "baro_alt"],
      windowS: 0,
      threshold: 3,
      measure: (w) => Math.abs(get(w.now, "alt") - get(w.now, "baro_alt")),
      message: (v) => `GPS altitude and barometer disagree by ${fmt(v, 1)} m`,
    },
    {
      rule: "groundspeed_vs_vel",
      channels: ["groundspeed", "vn", "ve"],
      windowS: 0,
      threshold: 1.5,
      measure: (w) => Math.abs(get(w.now, "groundspeed") - Math.hypot(get(w.now, "vn"), get(w.now, "ve"))),
      message: (v) => `Groundspeed doesn't match velocity components (off by ${fmt(v, 1)} m/s)`,
    },
    {
      rule: "vn_vs_dpn",
      channels: ["vn", "pn"],
      windowS: 1,
      threshold: 2,
      measure: (w) => Math.abs(w.mean("vn") - rate(w, "pn")),
      message: (v) => `North position moving ${fmt(v, 1)} m/s faster than reported velocity allows`,
    },
    {
      rule: "ve_vs_dpe",
      channels: ["ve", "pe"],
      windowS: 1,
      threshold: 2,
      measure: (w) => Math.abs(w.mean("ve") - rate(w, "pe")),
      message: (v) => `East position moving ${fmt(v, 1)} m/s faster than reported velocity allows`,
    },
  ],
  spacecraft: [
    {
      rule: "rate_limit",
      channels: ["value"],
      windowS: 0.15, // the previous frame at 10 Hz
      threshold: 0.15,
      measure: (w) => Math.abs(get(w.now, "value") - get(w.then, "value")),
      message: (v) => `Value jumped ${fmt(v, 2)} in one frame, beyond the channel's physical slew limit`,
    },
  ],
};

export const MOCK_THRESHOLD = 0.5;
const ALERT_ON_FRAMES = 3;
const ALERT_OFF_FRAMES = 25;

export class MockDetector {
  private history: { t: number; channels: Channels }[] = [];
  private above = 0;
  private below = 0;
  private alerting = false;

  constructor(
    public name: string,
    private readonly model: TruthModel,
    private readonly rng: Rng,
  ) {}

  reset(): void {
    this.history = [];
    this.above = 0;
    this.below = 0;
    this.alerting = false;
  }

  /** `attack` is only used to fake the base model's blind spot to the PPO attacker. */
  update(truth: Frame, reported: Frame, attack: string): Verdict {
    this.history.push({ t: reported.t, channels: reported.channels });
    const maxWindow = 5.5;
    while (this.history.length > 2 && reported.t - (this.history[0]?.t ?? reported.t) > maxWindow) this.history.shift();

    const violations = this.runL1(reported);
    const l1Score = violations.length === 0 ? 0 : Math.min(1, 0.6 + 0.2 * Math.max(...violations.map((v) => v.value / v.threshold - 1)));

    const blindSpot = attack === "ppo" && this.name !== "hardened";
    const residuals = this.residuals(truth.channels, reported.channels, blindSpot ? 0.012 : 1);
    const maxZ = Math.max(0, ...Object.values(residuals).map(Math.abs));
    const l2Score = 1 - Math.exp(-Math.max(0, maxZ - 2.5) / 1.5);

    const l3 = this.name === "hardened" ? this.discriminator(truth.channels, reported.channels) : null;
    const score = Math.max(l1Score, l2Score, l3 ?? 0);
    const alert = this.updateAlert(score);

    return {
      vehicle_id: reported.vehicle_id,
      seq: reported.seq,
      t: reported.t,
      l0: { sync_ok: true, crc_ok: true, seq_ok: true, time_ok: true, reasons: [], ok: true },
      l1_violations: violations,
      l1_score: round(l1Score),
      l2_residuals: residuals,
      l2_score: round(l2Score),
      l3_prob: l3 === null ? null : round(l3),
      score: round(score),
      alert,
      explanation: alert || score >= MOCK_THRESHOLD ? explain(violations, residuals, l3, l1Score, l2Score) : "",
      detector: this.name,
    };
  }

  private runL1(reported: Frame): Violation[] {
    const out: Violation[] = [];
    for (const rule of RULES[this.model.schema.name]) {
      const w = this.window(reported, rule.windowS);
      if (!w) continue;
      const value = rule.measure(w);
      if (value > rule.threshold) {
        out.push({
          rule: rule.rule,
          channels: rule.channels,
          value: round(value),
          threshold: rule.threshold,
          message: rule.message(value, w),
        });
      }
    }
    return out;
  }

  private window(reported: Frame, windowS: number): Window | null {
    const now = reported.channels;
    if (windowS === 0) return { now, then: now, dt: 1, mean: (k) => get(now, k) };
    const start = this.history.findIndex((h) => h.t >= reported.t - windowS - 1e-6);
    const first = this.history[start];
    if (start < 0 || !first || reported.t - first.t < windowS * 0.5) return null;
    const span = this.history.slice(start);
    return {
      now,
      then: first.channels,
      dt: reported.t - first.t,
      mean: (k) => span.reduce((s, h) => s + get(h.channels, k), 0) / span.length,
    };
  }

  private residuals(truth: Channels, reported: Channels, gain: number): Record<string, number> {
    const out: Record<string, number> = {};
    for (const [k, scale] of Object.entries(this.model.scales)) {
      const gap = (get(reported, k) - get(truth, k)) / scale;
      out[k] = round(gain * clamp(gap, -40, 40) + 0.7 * this.rng.gauss());
    }
    return out;
  }

  private discriminator(truth: Channels, reported: Channels): number {
    const gaps = Object.entries(this.model.scales).map(([k, s]) => ((get(reported, k) - get(truth, k)) / s) ** 2);
    const rms = Math.sqrt(gaps.reduce((a, b) => a + b, 0) / Math.max(1, gaps.length));
    const p = 1 / (1 + Math.exp(-(rms - 1.5) * 2.5));
    return clamp(p + 0.01 * this.rng.gauss(), 0.001, 0.999);
  }

  private updateAlert(score: number): boolean {
    if (score >= MOCK_THRESHOLD) {
      this.above += 1;
      this.below = 0;
    } else {
      this.below += 1;
      this.above = 0;
    }
    if (!this.alerting && this.above >= ALERT_ON_FRAMES) this.alerting = true;
    if (this.alerting && this.below >= ALERT_OFF_FRAMES) this.alerting = false;
    return this.alerting;
  }
}

function explain(
  violations: Violation[],
  residuals: Record<string, number>,
  l3: number | null,
  l1: number,
  l2: number,
): string {
  const suffix = " Link checks all pass: this frame is valid but wrong.";
  const first = violations[0];
  if (first && l1 >= l2 && l1 >= (l3 ?? 0)) return `${first.message}.${suffix}`;
  if (l3 !== null && l3 >= l2) {
    return `Hardened discriminator: P(forged) = ${l3.toFixed(2)}; frame pattern matches adversarial traffic.${suffix}`;
  }
  const [channel, z] = topResidual(residuals);
  return `${channel} is ${Math.abs(z).toFixed(1)}σ off the learned next-step prediction.${suffix}`;
}

export function topResidual(residuals: Record<string, number>): [string, number] {
  let best: [string, number] = ["", 0];
  for (const [k, z] of Object.entries(residuals)) if (Math.abs(z) > Math.abs(best[1])) best = [k, z];
  return best;
}

/** One line for the voice callout, mission-control style. */
export function spokenLine(verdict: Verdict): string {
  const confidence = Math.round(verdict.score * 100);
  const first = verdict.l1_violations[0];
  let what: string;
  if (first && verdict.l1_score >= verdict.l2_score) {
    what = `${first.channels.slice(0, 2).join(" and ")} physically inconsistent`;
  } else if (verdict.l3_prob !== null && verdict.l3_prob >= verdict.l2_score) {
    what = "hardened model flags forged telemetry";
  } else {
    what = `${topResidual(verdict.l2_residuals)[0]} channel diverging from predicted trajectory`;
  }
  return `Flight, GNC. ${capitalize(what)}. Confidence ${confidence} percent.`;
}

function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function round(v: number): number {
  return Math.round(v * 1000) / 1000;
}
