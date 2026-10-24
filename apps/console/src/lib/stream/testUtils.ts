import type { TickMessage, Verdict } from "@/lib/types";

/** Builds a contract-shaped tick for tests. */
export function makeTick(
  t: number,
  truth: Record<string, number>,
  reported: Record<string, number>,
  opts: { score?: number; alert?: boolean; active?: boolean; attack?: string; crcOk?: boolean; detector?: string } = {},
): TickMessage {
  const crcOk = opts.crcOk ?? true;
  const verdict: Verdict = {
    vehicle_id: "V",
    seq: Math.round(t * 10),
    t,
    l0: { sync_ok: true, crc_ok: crcOk, seq_ok: true, time_ok: true, reasons: [], ok: crcOk },
    l1_violations: [],
    l1_score: 0,
    l2_residuals: {},
    l2_score: 0,
    l3_prob: null,
    score: opts.score ?? 0,
    alert: opts.alert ?? false,
    explanation: opts.alert ? "explained" : "",
    detector: opts.detector ?? "base",
  };
  const frame = (channels: Record<string, number>) => ({ vehicle_id: "V", schema_id: 1, seq: verdict.seq, t, channels });
  return {
    type: "tick",
    truth: frame(truth),
    reported: frame(reported),
    verdict,
    attack: { name: opts.attack ?? (opts.active ? "bias" : "none"), active: opts.active ?? false },
  };
}
