export interface BackoffOptions {
  baseMs: number;
  maxMs: number;
  /** 0..1, fraction of the delay that is randomized so many consoles don't reconnect in lockstep. */
  jitter: number;
}

export const DEFAULT_BACKOFF: BackoffOptions = { baseMs: 500, maxMs: 8000, jitter: 0.2 };

/** Exponential backoff: base·2^attempt capped at max, with ±jitter. `attempt` starts at 0. */
export function backoffDelay(attempt: number, opts: BackoffOptions = DEFAULT_BACKOFF, random = Math.random): number {
  const exp = Math.min(opts.maxMs, opts.baseMs * 2 ** Math.max(0, attempt));
  const spread = exp * opts.jitter;
  return Math.round(Math.max(0, exp - spread + random() * 2 * spread));
}
