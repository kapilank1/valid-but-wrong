import { SCHEMA_IDS, type SchemaName, type ScenarioInfo } from "@/lib/types";

const SCHEMA_NAMES: readonly SchemaName[] = ["falcon9", "quad", "spacecraft"];

/**
 * Resolve which plot layout to use. The frame's schema_id is authoritative; the scenario's `schema`
 * field (name or id) and its id prefix are fallbacks for before the first tick arrives.
 */
export function resolveSchema(
  scenario: Pick<ScenarioInfo, "id" | "schema"> | null,
  frameSchemaId?: number,
): SchemaName | null {
  if (frameSchemaId !== undefined && SCHEMA_IDS[frameSchemaId]) return SCHEMA_IDS[frameSchemaId];
  if (!scenario) return null;
  const { schema, id } = scenario;
  if (typeof schema === "number" && SCHEMA_IDS[schema]) return SCHEMA_IDS[schema];
  if (typeof schema === "string" && (SCHEMA_NAMES as readonly string[]).includes(schema)) {
    return schema as SchemaName;
  }
  const prefix = id.split(":")[0];
  if (prefix === "falcon9") return "falcon9";
  if (prefix === "quad" || prefix === "sitl") return "quad";
  if (prefix === "smap" || prefix === "msl") return "spacecraft";
  return null;
}

export interface MissionEvent {
  name: string;
  t: number;
}

export interface TimeRange {
  start: number;
  end: number;
  label?: string;
}

/** Point events like {maxq: 81, meco: 161}, sorted by time. */
export function parseEvents(scenario: Pick<ScenarioInfo, "events"> | null): MissionEvent[] {
  const events = scenario?.events;
  if (!events || typeof events !== "object") return [];
  const out: MissionEvent[] = [];
  for (const [name, value] of Object.entries(events)) {
    if (typeof value === "number" && Number.isFinite(value)) out.push({ name, t: value });
  }
  return out.sort((a, b) => a.t - b.t);
}

function asRange(value: unknown, label?: string): TimeRange | null {
  if (Array.isArray(value) && value.length === 2) {
    const [a, b] = value;
    if (typeof a === "number" && typeof b === "number") return { start: Math.min(a, b), end: Math.max(a, b), label };
  }
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const v = value as Record<string, unknown>;
    const start = v.start ?? v.t0 ?? v.from;
    const end = v.end ?? v.t1 ?? v.to;
    if (typeof start === "number" && typeof end === "number") {
      const name = typeof v.label === "string" ? v.label : typeof v.class === "string" ? v.class : label;
      return { start: Math.min(start, end), end: Math.max(start, end), label: name };
    }
  }
  return null;
}

function asRangeList(value: unknown): TimeRange[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => asRange(v)).filter((r): r is TimeRange => r !== null);
}

/**
 * Labeled anomaly ranges (seconds of mission time) for spacecraft scenarios. The sources module owns
 * the exact shape, so accept the plausible ones: `scenario.anomalies` or `events.anomalies` as a list
 * of [start, end] pairs or {start, end} objects, or individual events whose value is a [start, end] pair.
 */
export function parseAnomalyRanges(scenario: Pick<ScenarioInfo, "events" | "anomalies"> | null): TimeRange[] {
  if (!scenario) return [];
  const ranges = [...asRangeList(scenario.anomalies)];
  const events = scenario.events;
  if (events && typeof events === "object") {
    for (const [name, value] of Object.entries(events)) {
      if (name === "anomalies" || name === "anomaly_ranges" || name === "anomaly_sequences") {
        ranges.push(...asRangeList(value));
        continue;
      }
      const single = asRange(value, name);
      if (single) ranges.push(single);
    }
  }
  return ranges.sort((a, b) => a.start - b.start);
}
