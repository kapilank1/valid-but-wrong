import { RingBuffer } from "@/lib/ringBuffer";
import type { TickMessage } from "@/lib/types";

/** Enough for a full Falcon 9 ascent at 20 Hz or a long spacecraft replay at 10 Hz. */
export const DEFAULT_CAPACITY = 20_000;

/**
 * Time-aligned history of every tick: truth and reported per channel plus detector outputs.
 * All series share one index space, so series[i] lines up with t[i].
 */
export class TelemetryBuffers {
  readonly t: RingBuffer;
  readonly score: RingBuffer;
  readonly alert: RingBuffer; // 1 while the verdict is alerting
  readonly attackActive: RingBuffer; // 1 while the attacker is rewriting frames (god view)
  readonly truth = new Map<string, RingBuffer>();
  readonly reported = new Map<string, RingBuffer>();
  /** Bumped on every mutation so canvases can skip redraws when nothing changed. */
  version = 0;

  constructor(readonly capacity: number = DEFAULT_CAPACITY) {
    this.t = new RingBuffer(capacity);
    this.score = new RingBuffer(capacity);
    this.alert = new RingBuffer(capacity);
    this.attackActive = new RingBuffer(capacity);
  }

  get length(): number {
    return this.t.length;
  }

  lastT(): number {
    return this.t.last();
  }

  push(tick: TickMessage): void {
    const { truth, reported, verdict, attack } = tick;
    this.t.push(reported.t);
    this.score.push(verdict.score);
    this.alert.push(verdict.alert ? 1 : 0);
    this.attackActive.push(attack.active ? 1 : 0);
    this.pushChannels(this.truth, truth.channels);
    this.pushChannels(this.reported, reported.channels);
    this.version += 1;
  }

  clear(): void {
    this.t.clear();
    this.score.clear();
    this.alert.clear();
    this.attackActive.clear();
    this.truth.clear();
    this.reported.clear();
    this.version += 1;
  }

  /** Index of the first sample at or after mission time `t`. */
  indexAtOrAfter(t: number): number {
    return this.t.lowerBound(t);
  }

  // Every channel buffer must grow by exactly one sample per tick so indices stay aligned with t:
  // missing channels get NaN, and a channel that first appears late is back-filled with NaN.
  private pushChannels(target: Map<string, RingBuffer>, channels: Record<string, number>): void {
    for (const [name, buf] of target) {
      const value = channels[name];
      buf.push(typeof value === "number" ? value : Number.NaN);
    }
    for (const [name, value] of Object.entries(channels)) {
      if (target.has(name)) continue;
      const buf = new RingBuffer(this.capacity);
      const backfill = Math.min(this.t.length - 1, this.capacity - 1);
      for (let i = 0; i < backfill; i++) buf.push(Number.NaN);
      buf.push(value);
      target.set(name, buf);
    }
  }
}
