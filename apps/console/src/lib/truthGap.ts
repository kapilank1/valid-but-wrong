import type { LinkCounters } from "@/lib/stream/reducer";
import type { SchemaName, TickMessage } from "@/lib/types";

/**
 * The two numbers the whole project is about, side by side: how much of the link is valid, and how far the
 * operator's picture is from the truth (god view: reported − true on the channel the attacker lies about).
 */

export type GapLevel = "none" | "small" | "large";

export interface TruthGap {
  value: number; // magnitude in `unit`
  text: string; // formatted magnitude, e.g. "3.72"
  unit: string;
  channel: string; // spoken name of what is off, e.g. "ALTITUDE"
  level: GapLevel;
}

interface GapSpec {
  channel: string;
  measure(d: (k: string) => number): number; // in base units (m / norm)
  small: number; // below this it's sensor-noise-sized: "none"
  large: number; // above this it's a lie that matters
  format(v: number): { text: string; unit: string };
}

const SPECS: Record<SchemaName, GapSpec> = {
  falcon9: {
    channel: "ALTITUDE",
    measure: (d) => Math.abs(d("altitude")),
    small: 25,
    large: 500,
    format: (m) => (m >= 1000 ? { text: (m / 1000).toFixed(2), unit: "km" } : { text: m.toFixed(0), unit: "m" }),
  },
  quad: {
    channel: "POSITION",
    measure: (d) => Math.hypot(d("pn"), d("pe"), d("alt")),
    small: 0.5,
    large: 5,
    format: (m) => ({ text: m >= 100 ? m.toFixed(0) : m.toFixed(1), unit: "m" }),
  },
  spacecraft: {
    channel: "CHANNEL",
    measure: (d) => Math.abs(d("value")),
    small: 0.02,
    large: 0.2,
    format: (v) => ({ text: v.toFixed(2), unit: "norm" }),
  },
};

export function truthGap(tick: TickMessage | null, schema: SchemaName): TruthGap | null {
  if (!tick) return null;
  const spec = SPECS[schema];
  const d = (k: string) => (tick.reported.channels[k] ?? Number.NaN) - (tick.truth.channels[k] ?? Number.NaN);
  const value = spec.measure(d);
  if (!Number.isFinite(value)) return null;
  const level: GapLevel = value < spec.small ? "none" : value < spec.large ? "small" : "large";
  return { value, ...spec.format(value), channel: spec.channel, level };
}

/** Altitude lie in km for Falcon 9 (signed), or null; the demo script watches this. */
export function altitudeLieKm(tick: TickMessage | null): number | null {
  if (!tick) return null;
  const v = (tick.reported.channels.altitude ?? Number.NaN) - (tick.truth.channels.altitude ?? Number.NaN);
  return Number.isFinite(v) ? v / 1000 : null;
}

export interface LinkValidity {
  fraction: number | null; // frames passing every L0 check / frames, null before the first frame
  text: string;
  forged: number;
  forgedPassing: number;
}

export function linkValidity(link: LinkCounters): LinkValidity {
  const fraction = link.frames === 0 ? null : link.allOk / link.frames;
  const text = fraction === null ? "—" : fraction === 1 ? "100%" : `${(fraction * 100).toFixed(1)}%`;
  return { fraction, text, forged: link.forged, forgedPassing: link.forgedPassing };
}
