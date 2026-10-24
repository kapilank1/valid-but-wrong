import { describe, expect, it } from "vitest";
import { makeTick } from "@/lib/stream/testUtils";
import { parseServerMessage } from "@/lib/transport/parse";

describe("parseServerMessage", () => {
  it("accepts contract messages", () => {
    const tick = makeTick(1, { a: 1 }, { a: 2 });
    expect(parseServerMessage(JSON.stringify(tick))).toEqual(tick);
    const alert = { type: "alert", verdict: tick.verdict, spoken: "hi" };
    expect(parseServerMessage(JSON.stringify(alert))).toEqual(alert);
  });

  it("rejects junk and unknown types", () => {
    expect(parseServerMessage("not json")).toBeNull();
    expect(parseServerMessage("[1,2]")).toBeNull();
    expect(parseServerMessage(JSON.stringify({ type: "nope" }))).toBeNull();
    expect(parseServerMessage(JSON.stringify({ type: "tick", truth: {} }))).toBeNull();
  });

  it("tolerates a tick without the attack field", () => {
    const { attack: _attack, ...rest } = makeTick(1, {}, {});
    const parsed = parseServerMessage(JSON.stringify(rest));
    expect(parsed?.type === "tick" && parsed.attack).toEqual({ name: "none", active: false });
  });
});
