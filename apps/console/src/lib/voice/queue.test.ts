import { describe, expect, it } from "vitest";
import { CalloutQueue } from "@/lib/voice/queue";

function deferredSpeaker() {
  const spoken: string[] = [];
  const pending: (() => void)[] = [];
  const speak = (text: string) =>
    new Promise<void>((resolve) => {
      spoken.push(text);
      pending.push(resolve);
    });
  const finishOne = async () => {
    pending.shift()?.();
    await new Promise((r) => setTimeout(r, 0));
  };
  return { spoken, speak, finishOne };
}

describe("CalloutQueue", () => {
  it("speaks one line at a time, in order", async () => {
    const s = deferredSpeaker();
    const q = new CalloutQueue(s.speak);
    q.enqueue("one");
    q.enqueue("two");
    expect(s.spoken).toEqual(["one"]);
    await s.finishOne();
    expect(s.spoken).toEqual(["one", "two"]);
    await s.finishOne();
    expect(q.speaking).toBe(false);
  });

  it("drops stale lines beyond maxPending and duplicates", async () => {
    const s = deferredSpeaker();
    const q = new CalloutQueue(s.speak, () => {}, 2);
    q.enqueue("a");
    ["b", "c", "c", "d"].forEach((t) => q.enqueue(t));
    expect(q.size).toBe(2);
    await s.finishOne();
    await s.finishOne();
    await s.finishOne();
    expect(s.spoken).toEqual(["a", "c", "d"]);
  });

  it("mute clears the queue, stops playback and ignores new lines", () => {
    const s = deferredSpeaker();
    let stopped = 0;
    const q = new CalloutQueue(s.speak, () => (stopped += 1));
    q.enqueue("a");
    q.enqueue("b");
    q.setMuted(true);
    q.enqueue("c");
    expect(stopped).toBe(1);
    expect(q.size).toBe(0);
    expect(s.spoken).toEqual(["a"]);
  });

  it("survives a failing speaker", async () => {
    const spoken: string[] = [];
    const q = new CalloutQueue(async (t) => {
      spoken.push(t);
      if (t === "bad") throw new Error("boom");
    });
    q.enqueue("bad");
    q.enqueue("good");
    await new Promise((r) => setTimeout(r, 0));
    expect(spoken).toEqual(["bad", "good"]);
  });
});
