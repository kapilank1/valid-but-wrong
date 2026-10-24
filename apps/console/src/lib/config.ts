/** Base URL of the pipeline server (FastAPI). Public by design: it holds no secrets. */
export const API_BASE = (process.env.NEXT_PUBLIC_VBW_API ?? "http://localhost:8000").replace(/\/+$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

/** ws(s)://host/ws derived from the REST base so one env var configures both. */
export function wsUrl(base: string = API_BASE): string {
  const url = new URL("/ws", base);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}

/** How long the initial health probe may take before we fall back to the mock stream. */
export const HEALTH_TIMEOUT_MS = 1500;
