/**
 * Voice callout proxy: POST {text} → audio/mpeg from Cartesia. Keeps CARTESIA_API_KEY server-side.
 *
 * Cartesia TTS REST reference: https://docs.cartesia.ai/api-reference/tts/bytes
 *   POST https://api.cartesia.ai/tts/bytes
 *   Authorization: Bearer <api key>, Cartesia-Version: 2026-08-14
 *   body {model_id, transcript, voice: {mode: "id", id}, output_format, language}
 * Model ids per that page: sonic-3.6 (default), sonic-3.5, sonic-3, sonic-latest.
 */
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CARTESIA_URL = "https://api.cartesia.ai/tts/bytes";
const CARTESIA_VERSION = process.env.CARTESIA_VERSION ?? "2026-08-14";
const MODEL_ID = process.env.CARTESIA_MODEL_ID ?? "sonic-3.6";
// Voice from Cartesia's own Next.js example (docs.cartesia.ai/examples/nextjs); override via env.
const VOICE_ID = process.env.CARTESIA_VOICE_ID ?? "6ccbfb76-1fc6-48f7-b71d-91ac6298247b";
const MAX_TEXT = 400;
const TIMEOUT_MS = 8000;

// Callouts repeat a lot during a demo; caching them cuts latency and API usage.
const CACHE_LIMIT = 64;
const cache = new Map<string, ArrayBuffer>();

function remember(text: string, audio: ArrayBuffer): void {
  cache.set(text, audio);
  if (cache.size > CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
}

function audioResponse(audio: ArrayBuffer, cached: boolean): Response {
  return new Response(audio, {
    headers: { "Content-Type": "audio/mpeg", "Cache-Control": "no-store", "X-TTS-Cache": cached ? "hit" : "miss" },
  });
}

/** Lets the client see whether Cartesia is configured without spending a synthesis call. */
export function GET() {
  return NextResponse.json({ configured: Boolean(process.env.CARTESIA_API_KEY), model: MODEL_ID });
}

export async function POST(request: Request) {
  const apiKey = process.env.CARTESIA_API_KEY;
  if (!apiKey) return NextResponse.json({ error: "tts_unconfigured" }, { status: 503 });

  let text: unknown;
  try {
    ({ text } = (await request.json()) as { text?: unknown });
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }
  if (typeof text !== "string" || text.trim() === "" || text.length > MAX_TEXT) {
    return NextResponse.json({ error: `text must be 1..${MAX_TEXT} characters` }, { status: 400 });
  }
  const transcript = text.trim();
  const hit = cache.get(transcript);
  if (hit) return audioResponse(hit, true);

  try {
    const upstream = await fetch(CARTESIA_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Cartesia-Version": CARTESIA_VERSION,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_id: MODEL_ID,
        transcript,
        voice: { mode: "id", id: VOICE_ID },
        output_format: { container: "mp3", sample_rate: 44100, bit_rate: 128000 },
        language: "en",
      }),
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (!upstream.ok) {
      const detail = (await upstream.text().catch(() => "")).slice(0, 300);
      return NextResponse.json({ error: "tts_upstream", status: upstream.status, detail }, { status: 502 });
    }
    const audio = await upstream.arrayBuffer();
    remember(transcript, audio);
    return audioResponse(audio, false);
  } catch (err) {
    const reason = err instanceof Error ? err.name : "unknown";
    return NextResponse.json({ error: "tts_unreachable", reason }, { status: 502 });
  }
}
