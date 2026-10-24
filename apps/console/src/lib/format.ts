/** Mission clock: "T+ 00:02:13.4" (or "T- …" before liftoff). */
export function formatMissionClock(t: number | null | undefined): string {
  if (t === null || t === undefined || !Number.isFinite(t)) return "T+ --:--:--.-";
  const sign = t < 0 ? "T-" : "T+";
  const tenths = Math.floor(Math.abs(t) * 10);
  const h = Math.floor(tenths / 36000);
  const m = Math.floor((tenths % 36000) / 600);
  const s = Math.floor((tenths % 600) / 10);
  const d = tenths % 10;
  return `${sign} ${pad(h)}:${pad(m)}:${pad(s)}.${d}`;
}

/** Short stamp for event logs: "T+02:13.4". */
export function formatStamp(t: number): string {
  if (!Number.isFinite(t)) return "T+--:--.-";
  const sign = t < 0 ? "T-" : "T+";
  const tenths = Math.floor(Math.abs(t) * 10);
  const m = Math.floor(tenths / 600);
  const s = Math.floor((tenths % 600) / 10);
  return `${sign}${pad(m)}:${pad(s)}.${tenths % 10}`;
}

function pad(n: number): string {
  return n.toString().padStart(2, "0");
}

/** Fixed-width-friendly number with a sensible number of decimals for its magnitude. */
export function formatValue(v: number | null | undefined, digits?: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const abs = Math.abs(v);
  const d = digits ?? (abs >= 1000 ? 0 : abs >= 100 ? 1 : abs >= 1 ? 2 : 3);
  return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
}

export function formatPercent(fraction: number, digits = 1): string {
  if (!Number.isFinite(fraction)) return "—";
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}
