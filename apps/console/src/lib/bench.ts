/**
 * Normalizes GET /api/bench (results/bench_summary.json) into rows the bench card can render.
 * The bench module owns that file's exact layout, so this accepts the plausible shapes:
 *   {datasets: {SMAP: prf, MSL: prf}, baseline: {name, SMAP: prf, MSL: prf}}
 *   {SMAP: {ours: prf, baseline: prf}, MSL: …, baseline_name}
 *   {results: [{dataset: "SMAP", precision, recall, f1, baseline: prf}]}
 * where prf = {precision|p, recall|r, f1|F1}, as fractions or percentages.
 */

export interface Prf {
  precision: number;
  recall: number;
  f1: number;
}

export interface BenchRow {
  dataset: string;
  ours: Prf;
  baseline: Prf | null;
}

export interface BenchView {
  rows: BenchRow[];
  baselineName: string | null;
  latencyMs: number | null;
  mock: boolean;
}

type Obj = Record<string, unknown>;

const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);

function pick(o: Obj, keys: string[]): unknown {
  for (const key of Object.keys(o)) {
    if (keys.includes(key.toLowerCase())) return o[key];
  }
  return undefined;
}

function fraction(v: unknown): number | null {
  const n = typeof v === "string" ? Number(v) : v;
  if (typeof n !== "number" || !Number.isFinite(n)) return null;
  return n > 1 ? n / 100 : n;
}

export function readPrf(v: unknown): Prf | null {
  if (!isObj(v)) return null;
  const precision = fraction(pick(v, ["precision", "p", "prec"]));
  const recall = fraction(pick(v, ["recall", "r", "rec"]));
  let f1 = fraction(pick(v, ["f1", "f1_score", "f"]));
  if (precision === null || recall === null) return null;
  if (f1 === null) f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
  return { precision, recall, f1 };
}

/** Case-insensitive lookup of a dataset entry in an object keyed by name, or a list of {dataset|name}. */
function findDataset(container: unknown, name: string): unknown {
  const lower = name.toLowerCase();
  if (Array.isArray(container)) {
    return container.find((e) => isObj(e) && String(e.dataset ?? e.name ?? "").toLowerCase() === lower);
  }
  if (isObj(container)) {
    const key = Object.keys(container).find((k) => k.toLowerCase() === lower);
    return key === undefined ? undefined : container[key];
  }
  return undefined;
}

function baselineContainer(raw: Obj): unknown {
  const b = raw.baseline ?? raw.baselines ?? raw.telemanom;
  return Array.isArray(b) ? b[0] : b;
}

function readLatency(raw: Obj): number | null {
  const v = raw.latency_ms ?? raw.detection_latency_ms ?? raw.latency;
  if (typeof v === "number") return v;
  if (isObj(v)) {
    const p = v.p50 ?? v.median ?? v.mean;
    return typeof p === "number" ? p : null;
  }
  return null;
}

export function normalizeBench(raw: unknown, datasets: string[] = ["SMAP", "MSL"]): BenchView | null {
  if (!isObj(raw) || raw.available === false) return null;
  const containers = [raw.datasets, raw.results, raw.ours, raw.metrics, raw];
  const baselines = baselineContainer(raw);
  const rows: BenchRow[] = [];

  for (const name of datasets) {
    let entry: unknown;
    for (const c of containers) {
      entry = findDataset(c, name);
      if (entry !== undefined) break;
    }
    if (!isObj(entry)) continue;
    const ours = readPrf(isObj(entry.ours) ? entry.ours : entry);
    if (!ours) continue;
    const baseline =
      readPrf(entry.baseline ?? entry.telemanom) ?? readPrf(findDataset(baselines, name)) ?? null;
    rows.push({ dataset: name, ours, baseline });
  }
  if (rows.length === 0) return null;

  const bl = baselineContainer(raw);
  const named = isObj(bl) && typeof bl.name === "string" ? bl.name : null;
  const baselineName =
    named ?? (typeof raw.baseline_name === "string" ? raw.baseline_name : rows.some((r) => r.baseline) ? "Telemanom" : null);
  return { rows, baselineName, latencyMs: readLatency(raw), mock: raw.mock === true };
}
