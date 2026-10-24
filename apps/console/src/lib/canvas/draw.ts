import type { RingBuffer } from "@/lib/ringBuffer";

export interface Theme {
  bg: string;
  panel: string;
  grid: string;
  axis: string;
  text: string;
  muted: string;
  truth: string;
  reported: string;
  alert: string;
  ok: string;
  mono: string;
}

let cachedTheme: Theme | null = null;

/** Canvas colors come from the same CSS custom properties as the DOM (see globals.css). */
export function readTheme(): Theme {
  if (cachedTheme) return cachedTheme;
  const css = getComputedStyle(document.documentElement);
  const v = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  cachedTheme = {
    bg: v("--color-bg", "#05070a"),
    panel: v("--color-panel", "#0a0d12"),
    grid: v("--color-grid", "#141a21"),
    axis: v("--color-line-strong", "#26303b"),
    text: v("--color-text", "#d5dde5"),
    muted: v("--color-muted", "#7d8a97"),
    truth: v("--color-truth", "#46d5e8"),
    reported: v("--color-reported", "#ffb020"),
    alert: v("--color-alert", "#ff3344"),
    ok: v("--color-ok", "#3ddc84"),
    mono: v("--font-mono", "ui-monospace, monospace"),
  };
  return cachedTheme;
}

export interface Surface {
  ctx: CanvasRenderingContext2D;
  width: number; // CSS pixels
  height: number;
  /** True when the backing store was resized this call, i.e. a redraw is required. */
  resized: boolean;
}

/** Match the canvas backing store to its CSS size × devicePixelRatio. */
export function prepare(canvas: HTMLCanvasElement): Surface | null {
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const bw = Math.max(1, Math.round(width * dpr));
  const bh = Math.max(1, Math.round(height * dpr));
  const resized = canvas.width !== bw || canvas.height !== bh;
  if (resized) {
    canvas.width = bw;
    canvas.height = bh;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height, resized };
}

/**
 * Stroke buffer samples [from, to) as one path. Long histories are strided to ~2 points per pixel;
 * NaN breaks the line.
 */
export function strokeSeries(
  ctx: CanvasRenderingContext2D,
  xs: RingBuffer,
  ys: RingBuffer,
  from: number,
  to: number,
  sx: (v: number) => number,
  sy: (v: number) => number,
  maxPoints: number,
  xScale = 1,
  yScale = 1,
): void {
  const n = to - from;
  if (n < 2) return;
  const stride = Math.max(1, Math.floor(n / Math.max(2, maxPoints)));
  ctx.beginPath();
  let pen = false;
  for (let i = from; i < to; i += stride) {
    const x = xs.get(i) * xScale;
    const y = ys.get(i) * yScale;
    if (Number.isNaN(x) || Number.isNaN(y)) {
      pen = false;
      continue;
    }
    if (pen) ctx.lineTo(sx(x), sy(y));
    else ctx.moveTo(sx(x), sy(y));
    pen = true;
  }
  // Always end on the newest sample so the line reaches the current-position marker.
  const lx = xs.get(to - 1) * xScale;
  const ly = ys.get(to - 1) * yScale;
  if (pen && !Number.isNaN(lx) && !Number.isNaN(ly)) ctx.lineTo(sx(lx), sy(ly));
  ctx.stroke();
}

export function font(theme: Theme, sizePx: number, weight = 400): string {
  return `${weight} ${sizePx}px ${theme.mono}`;
}

export function formatTick(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 10_000) return `${Math.round(v / 1000)}k`;
  if (abs >= 100 || Number.isInteger(v)) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(1);
  return v.toFixed(2);
}
