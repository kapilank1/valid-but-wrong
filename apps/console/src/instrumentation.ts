/** Next.js server start hook (runs once per server process, before any request). */
export async function register() {
  // Next inlines NEXT_RUNTIME at build time, so the Edge bundle drops this import (it needs node:tls).
  if (process.env.NEXT_RUNTIME === "nodejs") {
    await import("./instrumentation-node");
  }
}
