/**
 * Fixed-capacity numeric ring buffer. Index 0 is the oldest retained sample.
 * Lives outside React state so 10–20 Hz ticks never trigger re-renders.
 */
export class RingBuffer {
  private readonly data: Float64Array;
  private start = 0;
  private size = 0;

  constructor(readonly capacity: number) {
    if (!Number.isInteger(capacity) || capacity <= 0) {
      throw new Error(`RingBuffer capacity must be a positive integer, got ${capacity}`);
    }
    this.data = new Float64Array(capacity);
  }

  get length(): number {
    return this.size;
  }

  push(value: number): void {
    const end = (this.start + this.size) % this.capacity;
    this.data[end] = value;
    if (this.size < this.capacity) {
      this.size += 1;
    } else {
      this.start = (this.start + 1) % this.capacity;
    }
  }

  /** Sample i (0 = oldest). Returns NaN when out of range. */
  get(i: number): number {
    if (i < 0 || i >= this.size) return Number.NaN;
    return this.data[(this.start + i) % this.capacity] ?? Number.NaN;
  }

  last(): number {
    return this.get(this.size - 1);
  }

  clear(): void {
    this.start = 0;
    this.size = 0;
  }

  /**
   * First index whose value is >= target, assuming the buffer is sorted ascending (e.g. timestamps).
   * Returns length when every value is smaller.
   */
  lowerBound(target: number): number {
    let lo = 0;
    let hi = this.size;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.get(mid) < target) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  toArray(): number[] {
    const out = new Array<number>(this.size);
    for (let i = 0; i < this.size; i++) out[i] = this.get(i);
    return out;
  }
}
