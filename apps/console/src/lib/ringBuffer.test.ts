import { describe, expect, it } from "vitest";
import { RingBuffer } from "@/lib/ringBuffer";

describe("RingBuffer", () => {
  it("keeps samples in insertion order until full", () => {
    const b = new RingBuffer(4);
    [1, 2, 3].forEach((v) => b.push(v));
    expect(b.length).toBe(3);
    expect(b.toArray()).toEqual([1, 2, 3]);
    expect(b.last()).toBe(3);
  });

  it("overwrites the oldest sample when full", () => {
    const b = new RingBuffer(3);
    [1, 2, 3, 4, 5].forEach((v) => b.push(v));
    expect(b.length).toBe(3);
    expect(b.toArray()).toEqual([3, 4, 5]);
    expect(b.get(0)).toBe(3);
  });

  it("returns NaN out of range and empties on clear", () => {
    const b = new RingBuffer(2);
    expect(b.get(0)).toBeNaN();
    expect(b.last()).toBeNaN();
    b.push(7);
    expect(b.get(1)).toBeNaN();
    expect(b.get(-1)).toBeNaN();
    b.clear();
    expect(b.length).toBe(0);
  });

  it("binary-searches sorted timestamps across the wrap point", () => {
    const b = new RingBuffer(5);
    for (let t = 0; t < 8; t++) b.push(t); // holds 3..7
    expect(b.lowerBound(2)).toBe(0);
    expect(b.lowerBound(5)).toBe(2);
    expect(b.lowerBound(5.5)).toBe(3);
    expect(b.lowerBound(99)).toBe(5);
  });

  it("rejects bad capacities", () => {
    expect(() => new RingBuffer(0)).toThrow();
    expect(() => new RingBuffer(1.5)).toThrow();
  });
});
