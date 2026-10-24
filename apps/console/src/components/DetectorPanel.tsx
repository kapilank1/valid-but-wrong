"use client";

import { drawStrip, SCORE_KEY } from "@/lib/canvas/strip";
import { useCanvasDraw } from "@/lib/canvas/useCanvasDraw";
import { CHANNEL_DISPLAY, channelDisplay } from "@/lib/channels";
import type { TelemetryBuffers } from "@/lib/stream/buffers";
import type { SchemaName, Verdict } from "@/lib/types";
import { Panel } from "@/components/Panel";

interface DetectorPanelProps {
  buffers: TelemetryBuffers;
  verdict: Verdict | null;
  detector: string;
  threshold: number;
  schema: SchemaName;
}

export function DetectorPanel({ buffers, verdict, detector, threshold, schema }: DetectorPanelProps) {
  const score = verdict?.score ?? 0;
  const alert = verdict?.alert ?? false;
  return (
    <Panel
      title="Detector · L1 physics · L2 learned · L3"
      meta={
        <span className={`num border px-1.5 text-[10px] tracking-[0.12em] ${detector === "hardened" ? "border-text text-text" : "border-line-strong text-muted"}`}>
          {detector.toUpperCase()}
        </span>
      }
      bodyClassName="flex flex-col gap-3 overflow-hidden"
    >
      <div className="flex items-center gap-3">
        <ScoreGauge score={score} threshold={threshold} alert={alert} />
        <div className="flex flex-1 flex-col gap-1.5">
          <LayerBar label="L1 physics" value={verdict?.l1_score ?? 0} />
          <LayerBar label="L2 learned" value={verdict?.l2_score ?? 0} />
          <LayerBar label="L3 P(forged)" value={verdict?.l3_prob ?? null} />
        </div>
      </div>
      <ScoreHistory buffers={buffers} threshold={threshold} />
      <Violations verdict={verdict} />
      <Residuals verdict={verdict} schema={schema} />
    </Panel>
  );
}

// --------------------------------------------------------------------------- gauge

function ScoreGauge({ score, threshold, alert }: { score: number; threshold: number; alert: boolean }) {
  const r = 46;
  const cx = 60;
  const cy = 58;
  const point = (f: number) => {
    const a = Math.PI * (1 - f);
    return [cx + r * Math.cos(a), cy - r * Math.sin(a)] as const;
  };
  const arc = (f0: number, f1: number) => {
    const [x0, y0] = point(f0);
    const [x1, y1] = point(f1);
    return `M ${x0} ${y0} A ${r} ${r} 0 0 1 ${x1} ${y1}`;
  };
  const clamped = Math.min(1, Math.max(0, score));
  const over = alert || score >= threshold;
  const [tx0, ty0] = point(threshold);
  const tick = [cx + (r + 7) * Math.cos(Math.PI * (1 - threshold)), cy - (r + 7) * Math.sin(Math.PI * (1 - threshold))];

  return (
    <svg viewBox="0 0 120 70" className="h-[78px] w-[134px] flex-none" role="img" aria-label={`Combined score ${score.toFixed(2)}, threshold ${threshold}`}>
      <path d={arc(0, 1)} fill="none" stroke="var(--color-line-strong)" strokeWidth="7" />
      {clamped > 0.002 ? (
        <path d={arc(0, clamped)} fill="none" stroke={over ? "var(--color-alert)" : "var(--color-text)"} strokeWidth="7" />
      ) : null}
      <line x1={tx0} y1={ty0} x2={tick[0]} y2={tick[1]} stroke="var(--color-muted)" strokeWidth="1.5" />
      <text x={cx} y={cy - 6} textAnchor="middle" className="num" fontSize="22" fill={over ? "var(--color-alert)" : "var(--color-text)"}>
        {score.toFixed(2)}
      </text>
      <text x={cx} y={cy + 9} textAnchor="middle" className="num" fontSize="7.5" letterSpacing="1.2" fill="var(--color-muted)">
        THRESH {threshold.toFixed(2)}
      </text>
    </svg>
  );
}

function LayerBar({ label, value }: { label: string; value: number | null }) {
  const missing = value === null;
  return (
    <div className="grid grid-cols-[86px_1fr_34px] items-center gap-2">
      <span className="label text-[9.5px]">{label}</span>
      <div className="h-[5px] bg-line-strong/60">
        {!missing ? <div className="h-full bg-text/80" style={{ width: `${Math.min(1, Math.max(0, value)) * 100}%` }} /> : null}
      </div>
      <span className="num text-right text-[11px] text-muted">{missing ? "n/a" : value.toFixed(2)}</span>
    </div>
  );
}

function ScoreHistory({ buffers, threshold }: { buffers: TelemetryBuffers; threshold: number }) {
  const ref = useCanvasDraw(
    buffers,
    (surface, theme) => drawStrip(surface, theme, buffers, { channel: SCORE_KEY, scale: 1, threshold }),
    `score:${threshold}`,
  );
  return (
    <div>
      <div className="label mb-1">Combined score · last 60 s</div>
      <canvas ref={ref} className="h-[54px] w-full" aria-label="Score history" />
    </div>
  );
}

// --------------------------------------------------------------------------- L1 / L2

function Violations({ verdict }: { verdict: Verdict | null }) {
  const list = verdict?.l1_violations ?? [];
  return (
    <div className="min-h-[64px]">
      <div className="label mb-1">L1 physics violations {list.length > 0 ? `· ${list.length}` : ""}</div>
      {list.length === 0 ? (
        <div className="num text-[11px] text-faint">none: every frame is self-consistent</div>
      ) : (
        <ul className="flex flex-col gap-1.5">
          {list.slice(0, 3).map((v) => (
            <li key={v.rule} className="border-l-2 border-alert pl-2">
              <div className="num text-[10.5px] text-alert">
                {v.rule} <span className="text-muted">{v.value.toFixed(1)} &gt; {v.threshold}</span>
              </div>
              <div className="text-[11.5px] leading-snug text-text">{v.message}</div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

const Z_MAX = 8;
const Z_GUIDE = 3;
const MAX_BARS = 8;

function Residuals({ verdict, schema }: { verdict: Verdict | null; schema: SchemaName }) {
  const residuals = verdict?.l2_residuals ?? {};
  const rows = pickResiduals(residuals, schema);
  return (
    <div className="min-h-0 flex-1">
      <div className="label mb-1 flex justify-between">
        <span>L2 residuals · robust z</span>
        <span className="text-faint">|z| 0–{Z_MAX}σ</span>
      </div>
      {rows.length === 0 ? (
        <div className="num text-[11px] text-faint">no residuals yet</div>
      ) : (
        <ul className="flex flex-col gap-[5px]">
          {rows.map(([key, z]) => {
            const mag = Math.min(Z_MAX, Math.abs(z));
            const hot = Math.abs(z) >= Z_GUIDE;
            return (
              <li key={key} className="grid grid-cols-[82px_1fr_38px] items-center gap-2">
                <span className="num truncate text-[10px] text-muted">{channelDisplay(schema, key).label}</span>
                <div className="relative h-[7px] bg-line">
                  <div className={`h-full ${hot ? "bg-reported" : "bg-muted/70"}`} style={{ width: `${(mag / Z_MAX) * 100}%` }} />
                  <i className="absolute top-[-2px] h-[11px] w-px bg-faint" style={{ left: `${(Z_GUIDE / Z_MAX) * 100}%` }} />
                </div>
                <span className={`num text-right text-[10.5px] ${hot ? "text-reported" : "text-muted"}`}>{z.toFixed(1)}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/** Stable display order; when there are too many channels, keep the largest |z| but don't reshuffle them. */
function pickResiduals(residuals: Record<string, number>, schema: SchemaName): [string, number][] {
  const order = CHANNEL_DISPLAY[schema].map((c) => c.key);
  const rank = (k: string) => {
    const i = order.indexOf(k);
    return i < 0 ? order.length : i;
  };
  let rows = Object.entries(residuals).filter(([, z]) => Number.isFinite(z));
  if (rows.length > MAX_BARS) {
    rows = [...rows].sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, MAX_BARS);
  }
  return rows.sort((a, b) => rank(a[0]) - rank(b[0]));
}
