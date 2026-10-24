import { font, formatTick, strokeSeries, type Surface, type Theme } from "@/lib/canvas/draw";
import { Extent, linear, niceDomain, niceTicks, type Domain } from "@/lib/canvas/scale";
import { channelDisplay } from "@/lib/channels";
import type { RingBuffer } from "@/lib/ringBuffer";
import type { MissionEvent, TimeRange } from "@/lib/scenario";
import type { TelemetryBuffers } from "@/lib/stream/buffers";
import type { SchemaName } from "@/lib/types";

/** What the hero plot shows for each vehicle type. */
export interface HeroSpec {
  mode: "xy" | "time";
  x: string; // channel key, or "t" in time mode
  y: string;
  xScale: number; // raw → display units
  yScale: number;
  xTitle: string;
  yTitle: string;
  minSpan: [number, number];
  equalAspect: boolean;
  includeZero: boolean;
}

export function heroSpec(schema: SchemaName): HeroSpec {
  if (schema === "falcon9") {
    return {
      mode: "xy", x: "downrange", y: "altitude", xScale: 1e-3, yScale: 1e-3,
      xTitle: "DOWNRANGE km", yTitle: "ALTITUDE km", minSpan: [40, 20], equalAspect: false, includeZero: true,
    };
  }
  if (schema === "quad") {
    return {
      mode: "xy", x: "pe", y: "pn", xScale: 1, yScale: 1,
      xTitle: "EAST m", yTitle: "NORTH m", minSpan: [40, 40], equalAspect: true, includeZero: false,
    };
  }
  const value = channelDisplay("spacecraft", "value");
  return {
    mode: "time", x: "t", y: "value", xScale: 1, yScale: value.scale,
    xTitle: "MISSION TIME s", yTitle: "VALUE norm", minSpan: [60, 0.4], equalAspect: false, includeZero: false,
  };
}

export interface HeroOverlay {
  events: MissionEvent[];
  anomalies: TimeRange[];
  alertActive: boolean;
}

const M = { left: 48, right: 14, top: 12, bottom: 26 };
const MAX_POINTS = 3000;

export function drawHero(s: Surface, theme: Theme, buffers: TelemetryBuffers, spec: HeroSpec, overlay: HeroOverlay): void {
  const { ctx, width, height } = s;
  const plot = { x0: M.left, y0: M.top, x1: width - M.right, y1: height - M.bottom };
  if (plot.x1 - plot.x0 < 40 || plot.y1 - plot.y0 < 40) return;

  const xs = spec.mode === "time" ? buffers.t : buffers.truth.get(spec.x);
  const xr = spec.mode === "time" ? buffers.t : buffers.reported.get(spec.x);
  const yt = buffers.truth.get(spec.y);
  const yr = buffers.reported.get(spec.y);
  const n = buffers.length;

  const [xDomain, yDomain] = domains(spec, plot, n > 0 && xs && xr && yt && yr ? [xs, xr] : [], yt && yr ? [yt, yr] : []);
  const sx = linear(xDomain, [plot.x0, plot.x1]);
  const sy = linear(yDomain, [plot.y1, plot.y0]);
  const xMul = spec.mode === "time" ? 1 : spec.xScale;

  if (spec.mode === "time") drawRanges(ctx, theme, overlay.anomalies, sx, plot);
  drawGrid(ctx, theme, xDomain, yDomain, sx, sy, plot, spec);
  if (!xs || !xr || !yt || !yr || n < 1) {
    drawEmpty(ctx, theme, plot);
    return;
  }

  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x0, plot.y0, plot.x1 - plot.x0, plot.y1 - plot.y0);
  ctx.clip();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  if (spec.mode === "time") drawAlertBand(ctx, theme, buffers, sx, plot);

  ctx.lineWidth = 2;
  ctx.strokeStyle = theme.truth;
  ctx.shadowColor = theme.truth;
  ctx.shadowBlur = 6;
  strokeSeries(ctx, xs, yt, 0, n, sx, sy, MAX_POINTS, xMul, spec.yScale);
  ctx.shadowBlur = 0;

  ctx.lineWidth = 1.6;
  ctx.strokeStyle = theme.reported;
  strokeSeries(ctx, xr, yr, 0, n, sx, sy, MAX_POINTS, xMul, spec.yScale);

  if (spec.mode === "xy") {
    drawEvents(ctx, theme, buffers, xs, yt, overlay.events, sx, sy, spec);
    drawOnset(ctx, theme, buffers, xs, yt, sx, sy, spec);
  }
  drawHeads(ctx, theme, [sx(xs.last() * xMul), sy(yt.last() * spec.yScale)], [sx(xr.last() * xMul), sy(yr.last() * spec.yScale)], overlay.alertActive);
  ctx.restore();
}

function domains(spec: HeroSpec, plot: { x0: number; x1: number; y0: number; y1: number }, xBufs: RingBuffer[], yBufs: RingBuffer[]): [Domain, Domain] {
  const ex = new Extent();
  const ey = new Extent();
  const xMul = spec.mode === "time" ? 1 : spec.xScale;
  for (const b of xBufs) for (let i = 0; i < b.length; i++) ex.add(b.get(i) * xMul);
  for (const b of yBufs) for (let i = 0; i < b.length; i++) ey.add(b.get(i) * spec.yScale);
  if (spec.includeZero) {
    ex.add(0);
    ey.add(0);
  }
  if (ex.empty) ex.add(0);
  if (ey.empty) ey.add(0);
  let [xMin, xMax] = [ex.min, Math.max(ex.max, ex.min + spec.minSpan[0])];
  let [yMin, yMax] = [ey.min, Math.max(ey.max, ey.min + spec.minSpan[1])];
  if (spec.includeZero) {
    // Leave headroom so the vehicle never sits on the plot edge.
    xMax = xMin + (xMax - xMin) * 1.08;
    yMax = yMin + (yMax - yMin) * 1.12;
  }
  if (spec.equalAspect) {
    const pw = plot.x1 - plot.x0;
    const ph = plot.y1 - plot.y0;
    const unitsPerPx = Math.max((xMax - xMin) / pw, (yMax - yMin) / ph) * 1.1;
    const cx = (xMin + xMax) / 2;
    const cy = (yMin + yMax) / 2;
    [xMin, xMax] = [cx - (unitsPerPx * pw) / 2, cx + (unitsPerPx * pw) / 2];
    [yMin, yMax] = [cy - (unitsPerPx * ph) / 2, cy + (unitsPerPx * ph) / 2];
    return [[xMin, xMax], [yMin, yMax]];
  }
  if (!spec.includeZero) {
    const padY = (yMax - yMin) * 0.08;
    yMin -= padY;
    yMax += padY;
  }
  return [niceDomain(xMin, xMax, spec.minSpan[0], 8), niceDomain(yMin, yMax, spec.minSpan[1], 5)];
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  theme: Theme,
  xd: Domain,
  yd: Domain,
  sx: (v: number) => number,
  sy: (v: number) => number,
  plot: { x0: number; x1: number; y0: number; y1: number },
  spec: HeroSpec,
): void {
  ctx.save();
  ctx.lineWidth = 1;
  ctx.font = font(theme, 10);
  ctx.fillStyle = theme.muted;
  ctx.strokeStyle = theme.grid;

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const v of niceTicks(xd[0], xd[1], 8)) {
    const x = Math.round(sx(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, plot.y0);
    ctx.lineTo(x, plot.y1);
    ctx.stroke();
    ctx.fillText(formatTick(v), x, plot.y1 + 6);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const v of niceTicks(yd[0], yd[1], 5)) {
    const y = Math.round(sy(v)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(plot.x0, y);
    ctx.lineTo(plot.x1, y);
    ctx.stroke();
    ctx.fillText(formatTick(v), plot.x0 - 8, y);
  }

  ctx.strokeStyle = theme.axis;
  ctx.strokeRect(plot.x0 + 0.5, plot.y0 + 0.5, plot.x1 - plot.x0 - 1, plot.y1 - plot.y0 - 1);

  ctx.fillStyle = theme.muted;
  ctx.font = font(theme, 9.5);
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(spec.xTitle, plot.x1 - 6, plot.y1 - 5);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(spec.yTitle, plot.x0 + 6, plot.y0 + 6);
  ctx.restore();
}

function drawEmpty(ctx: CanvasRenderingContext2D, theme: Theme, plot: { x0: number; x1: number; y0: number; y1: number }): void {
  ctx.save();
  ctx.fillStyle = theme.muted;
  ctx.font = font(theme, 11);
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("AWAITING TELEMETRY · PRESS START", (plot.x0 + plot.x1) / 2, (plot.y0 + plot.y1) / 2);
  ctx.restore();
}

function drawRanges(
  ctx: CanvasRenderingContext2D,
  theme: Theme,
  ranges: TimeRange[],
  sx: (v: number) => number,
  plot: { x0: number; x1: number; y0: number; y1: number },
): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x0, plot.y0, plot.x1 - plot.x0, plot.y1 - plot.y0);
  ctx.clip();
  for (const r of ranges) {
    const a = sx(r.start);
    const b = sx(r.end);
    if (b < plot.x0 || a > plot.x1) continue;
    ctx.fillStyle = "rgba(210, 220, 230, 0.07)";
    ctx.fillRect(a, plot.y0, Math.max(1, b - a), plot.y1 - plot.y0);
    ctx.strokeStyle = "rgba(210, 220, 230, 0.22)";
    ctx.setLineDash([3, 3]);
    ctx.strokeRect(a + 0.5, plot.y0 + 0.5, Math.max(1, b - a), plot.y1 - plot.y0 - 1);
    ctx.setLineDash([]);
    ctx.fillStyle = theme.muted;
    ctx.font = font(theme, 9);
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("LABELED ANOMALY", a + 4, plot.y0 + 22);
  }
  ctx.restore();
}

/** Thin red band along the bottom while the detector was alerting (time mode only). */
function drawAlertBand(
  ctx: CanvasRenderingContext2D,
  theme: Theme,
  buffers: TelemetryBuffers,
  sx: (v: number) => number,
  plot: { x0: number; x1: number; y0: number; y1: number },
): void {
  ctx.fillStyle = theme.alert;
  let start = -1;
  const n = buffers.length;
  for (let i = 0; i <= n; i++) {
    const on = i < n && buffers.alert.get(i) === 1;
    if (on && start < 0) start = i;
    if (!on && start >= 0) {
      const a = sx(buffers.t.get(start));
      const b = sx(buffers.t.get(i - 1));
      ctx.fillRect(a, plot.y1 - 4, Math.max(2, b - a), 4);
      start = -1;
    }
  }
}

function drawEvents(
  ctx: CanvasRenderingContext2D,
  theme: Theme,
  buffers: TelemetryBuffers,
  xs: RingBuffer,
  ys: RingBuffer,
  events: MissionEvent[],
  sx: (v: number) => number,
  sy: (v: number) => number,
  spec: HeroSpec,
): void {
  const lastT = buffers.lastT();
  ctx.font = font(theme, 9);
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  for (const e of events) {
    if (!(e.t <= lastT)) continue;
    const i = buffers.indexAtOrAfter(e.t);
    if (i >= buffers.length) continue;
    const x = sx(xs.get(i) * spec.xScale);
    const y = sy(ys.get(i) * spec.yScale);
    if (Number.isNaN(x) || Number.isNaN(y)) continue;
    // Ascent curves are concave-down, so below-right of the track is empty space for labels.
    ctx.strokeStyle = theme.muted;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + 8, y + 12);
    ctx.stroke();
    ctx.fillStyle = theme.muted;
    ctx.fillText(e.name.replace(/_/g, " ").toUpperCase(), x + 11, y + 16);
  }
}

/** God view: mark where the attacker started rewriting frames. */
function drawOnset(
  ctx: CanvasRenderingContext2D,
  theme: Theme,
  buffers: TelemetryBuffers,
  xs: RingBuffer,
  ys: RingBuffer,
  sx: (v: number) => number,
  sy: (v: number) => number,
  spec: HeroSpec,
): void {
  const n = buffers.length;
  if (n === 0 || buffers.attackActive.get(n - 1) !== 1) return;
  let i = n - 1;
  while (i > 0 && buffers.attackActive.get(i - 1) === 1) i -= 1;
  const x = sx(xs.get(i) * spec.xScale);
  const y = sy(ys.get(i) * spec.yScale);
  if (Number.isNaN(x) || Number.isNaN(y)) return;
  ctx.strokeStyle = theme.reported;
  ctx.fillStyle = theme.bg;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = theme.reported;
  ctx.font = font(theme, 9, 600);
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("FORGERY BEGINS", x + 8, y + 6);
}

function drawHeads(ctx: CanvasRenderingContext2D, theme: Theme, truth: [number, number], reported: [number, number], alert: boolean): void {
  const [tx, ty] = truth;
  const [rx, ry] = reported;
  if ([tx, ty, rx, ry].some(Number.isNaN)) return;
  const gap = Math.hypot(rx - tx, ry - ty);
  if (gap > 6) {
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = alert ? theme.alert : theme.muted;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(rx, ry);
    ctx.stroke();
    ctx.restore();
  }
  ctx.lineWidth = 2;
  ctx.strokeStyle = theme.truth;
  ctx.fillStyle = theme.bg;
  ctx.beginPath();
  ctx.arc(tx, ty, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  if (alert) {
    ctx.strokeStyle = theme.alert;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(rx, ry, 11, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.fillStyle = theme.reported;
  ctx.beginPath();
  ctx.arc(rx, ry, 4, 0, Math.PI * 2);
  ctx.fill();
}
