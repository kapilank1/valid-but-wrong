export type Domain = [number, number];

/** Round tick positions ("1, 2, 5 × 10^k" steps) covering [min, max]. */
export function niceTicks(min: number, max: number, target = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const lo = Math.min(min, max);
  const hi = Math.max(min, max);
  const step = niceStep((hi - lo) / Math.max(1, target));
  const first = Math.ceil(lo / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let v = first; v <= hi + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toPrecision(12)));
  }
  return ticks;
}

export function niceStep(raw: number): number {
  if (!(raw > 0) || !Number.isFinite(raw)) return 1;
  const exp = Math.floor(Math.log10(raw));
  const base = 10 ** exp;
  const f = raw / base;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return nice * base;
}

/** Expand [min, max] outward to whole nice steps, with a minimum span so flat signals stay readable. */
export function niceDomain(min: number, max: number, minSpan = 1e-6, target = 5): Domain {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  let lo = min;
  let hi = max;
  if (hi - lo < minSpan) {
    const mid = (hi + lo) / 2;
    lo = mid - minSpan / 2;
    hi = mid + minSpan / 2;
  }
  const step = niceStep((hi - lo) / target);
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

export function linear(domain: Domain, range: Domain): (v: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

/** Running min/max that ignores NaN. */
export class Extent {
  min = Number.POSITIVE_INFINITY;
  max = Number.NEGATIVE_INFINITY;

  add(v: number): void {
    if (Number.isNaN(v)) return;
    if (v < this.min) this.min = v;
    if (v > this.max) this.max = v;
  }

  get empty(): boolean {
    return this.min > this.max;
  }
}
