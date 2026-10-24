import { font, formatTick, strokeSeries, type Surface, type Theme } from "@/lib/canvas/draw";
import { Extent, linear, niceDomain, niceTicks } from "@/lib/canvas/scale";
import type { RingBuffer } from "@/lib/ringBuffer";
import type { TelemetryBuffers } from "@/lib/stream/buffers";

export const STRIP_WINDOW_S = 60;
export const SCORE_KEY = "__score";

const M = { left: 34, right: 6, top: 4, bottom: 4 };

export interface StripOptions {
  channel: string; // channel key, or SCORE_KEY for the detector score
  scale: number;
  threshold?: number; // drawn as a dashed line for the score strip
}

/** Rolling-window small multiple: truth vs reported for one channel, gap between them filled. */
export function drawStrip(s: Surface, theme: Theme, buffers: TelemetryBuffers, opts: StripOptions): void {
  const { ctx, width, height } = s;
  const plot = { x0: M.left, y0: M.top, x1: width - M.right, y1: height - M.bottom };
  if (plot.x1 - plot.x0 < 20 || plot.y1 - plot.y0 < 12) return;
  const n = buffers.length;
  const lastT = buffers.lastT();
  const isScore = opts.channel === SCORE_KEY;
  const truth = isScore ? undefined : buffers.truth.get(opts.channel);
  const reported = isScore ? buffers.score : buffers.reported.get(opts.channel);

  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(plot.x0 + 0.5, plot.y0 + 0.5, plot.x1 - plot.x0 - 1, plot.y1 - plot.y0 - 1);
  if (n < 2 || !reported || Number.isNaN(lastT)) return;

  const t1 = Math.max(lastT, STRIP_WINDOW_S);
  const t0 = t1 - STRIP_WINDOW_S;
  const from = Math.max(0, buffers.indexAtOrAfter(t0) - 1);
  const sx = linear([t0, t1], [plot.x0, plot.x1]);

  const ext = new Extent();
  for (let i = from; i < n; i++) {
    ext.add(reported.get(i) * opts.scale);
    if (truth) ext.add(truth.get(i) * opts.scale);
  }
  if (ext.empty) return;
  const mag = Math.max(Math.abs(ext.min), Math.abs(ext.max));
  const [lo, hi] = isScore ? [0, 1] : niceDomain(ext.min, ext.max, Math.max(mag * 0.01, 1e-3), 3);
  const sy = linear([lo, hi], [plot.y1 - 2, plot.y0 + 2]);

  shadeFlags(ctx, buffers, buffers.attackActive, from, sx, plot, "rgba(255, 173, 31, 0.06)", plot.y0, plot.y1 - plot.y0);
  shadeFlags(ctx, buffers, buffers.alert, from, sx, plot, theme.alert, plot.y1 - 2, 2);

  ctx.font = font(theme, 9);
  ctx.fillStyle = theme.muted;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of niceTicks(lo, hi, 2)) {
    const y = Math.round(sy(v)) + 0.5;
    ctx.strokeStyle = theme.grid;
    ctx.beginPath();
    ctx.moveTo(plot.x0, y);
    ctx.lineTo(plot.x1, y);
    ctx.stroke();
    ctx.fillText(formatTick(v), plot.x0 - 4, y);
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x0, plot.y0, plot.x1 - plot.x0, plot.y1 - plot.y0);
  ctx.clip();
  ctx.lineJoin = "round";
  if (truth) {
    fillGap(ctx, buffers.t, truth, reported, from, n, sx, sy, opts.scale);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = theme.truth;
    strokeSeries(ctx, buffers.t, truth, from, n, sx, sy, 600, 1, opts.scale);
  }
  if (opts.threshold !== undefined) {
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = theme.muted;
    ctx.lineWidth = 1;
    const y = Math.round(sy(opts.threshold)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(plot.x0, y);
    ctx.lineTo(plot.x1, y);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.lineWidth = 1.3;
  ctx.strokeStyle = isScore ? theme.text : theme.reported;
  strokeSeries(ctx, buffers.t, reported, from, n, sx, sy, 600, 1, opts.scale);
  ctx.restore();
}

/** Translucent band between truth and reported, so even a small lie reads as a shape. */
function fillGap(
  ctx: CanvasRenderingContext2D,
  t: RingBuffer,
  truth: RingBuffer,
  reported: RingBuffer,
  from: number,
  to: number,
  sx: (v: number) => number,
  sy: (v: number) => number,
  scale: number,
): void {
  const stride = Math.max(1, Math.floor((to - from) / 400));
  ctx.beginPath();
  let started = false;
  for (let i = from; i < to; i += stride) {
    const x = sx(t.get(i));
    const y = sy(reported.get(i) * scale);
    if (Number.isNaN(y)) continue;
    if (started) ctx.lineTo(x, y);
    else ctx.moveTo(x, y);
    started = true;
  }
  if (!started) return;
  for (let i = to - 1; i >= from; i -= stride) {
    const y = sy(truth.get(i) * scale);
    if (!Number.isNaN(y)) ctx.lineTo(sx(t.get(i)), y);
  }
  ctx.closePath();
  ctx.fillStyle = "rgba(255, 173, 31, 0.22)";
  ctx.fill();
}

function shadeFlags(
  ctx: CanvasRenderingContext2D,
  buffers: TelemetryBuffers,
  flags: RingBuffer,
  from: number,
  sx: (v: number) => number,
  plot: { x0: number; x1: number },
  color: string,
  y: number,
  h: number,
): void {
  const n = buffers.length;
  ctx.fillStyle = color;
  let start = -1;
  for (let i = from; i <= n; i++) {
    const on = i < n && flags.get(i) === 1;
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      const a = Math.max(plot.x0, sx(buffers.t.get(start)));
      const b = Math.min(plot.x1, sx(buffers.t.get(i - 1)));
      if (b > a) ctx.fillRect(a, y, b - a, h);
      start = -1;
    }
  }
}
