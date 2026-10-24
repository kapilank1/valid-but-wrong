import type { ServerMessage } from "@/lib/types";

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function isFrame(v: unknown): boolean {
  return isObject(v) && typeof v.t === "number" && isObject(v.channels);
}

function isVerdict(v: unknown): boolean {
  return isObject(v) && typeof v.score === "number" && isObject(v.l0);
}

/**
 * Parse one WebSocket payload. Returns null for anything that doesn't look like a contract message,
 * so a malformed frame from a half-finished server build can't crash the console.
 */
export function parseServerMessage(raw: string): ServerMessage | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isObject(data)) return null;
  switch (data.type) {
    case "hello":
      return isObject(data.catalog) && isObject(data.status) ? (data as unknown as ServerMessage) : null;
    case "scenario":
      return isObject(data.scenario) ? (data as unknown as ServerMessage) : null;
    case "status":
      return isObject(data.status) ? (data as unknown as ServerMessage) : null;
    case "tick":
      if (!isFrame(data.truth) || !isFrame(data.reported) || !isVerdict(data.verdict)) return null;
      return {
        ...(data as unknown as Extract<ServerMessage, { type: "tick" }>),
        // `attack` is required by the contract; tolerate its absence rather than dropping the tick.
        attack: isObject(data.attack)
          ? { name: String(data.attack.name ?? "none"), active: Boolean(data.attack.active) }
          : { name: "none", active: false },
      };
    case "alert":
      return isVerdict(data.verdict)
        ? { type: "alert", verdict: data.verdict as never, spoken: typeof data.spoken === "string" ? data.spoken : "" }
        : null;
    default:
      return null;
  }
}
