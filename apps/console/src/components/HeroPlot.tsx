"use client";

import { useMemo } from "react";
import { drawHero, heroSpec } from "@/lib/canvas/hero";
import { useCanvasDraw } from "@/lib/canvas/useCanvasDraw";
import { formatValue } from "@/lib/format";
import { parseAnomalyRanges, parseEvents } from "@/lib/scenario";
import type { TelemetryBuffers } from "@/lib/stream/buffers";
import type { ScenarioInfo, SchemaName, TickMessage } from "@/lib/types";
import { Panel } from "@/components/Panel";

interface HeroPlotProps {
  buffers: TelemetryBuffers;
  schema: SchemaName;
  scenario: ScenarioInfo | null;
  latest: TickMessage | null;
  alertActive: boolean;
}

const TITLES: Record<SchemaName, string> = {
  falcon9: "Trajectory · altitude vs downrange",
  quad: "Ground track · north / east",
  spacecraft: "Channel value vs mission time",
};

export function HeroPlot({ buffers, schema, scenario, latest, alertActive }: HeroPlotProps) {
  const spec = useMemo(() => heroSpec(schema), [schema]);
  const overlay = useMemo(
    () => ({ events: parseEvents(scenario), anomalies: parseAnomalyRanges(scenario), alertActive }),
    [scenario, alertActive],
  );
  const canvasRef = useCanvasDraw(
    buffers,
    (surface, theme) => drawHero(surface, theme, buffers, spec, overlay),
    `${schema}|${scenario?.id ?? ""}|${alertActive}`,
  );
  const divergence = useMemo(() => measureDivergence(latest, schema), [latest, schema]);

  return (
    <Panel
      title={TITLES[schema]}
      meta={<Legend />}
      bodyClassName="relative p-0"
    >
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-label="Truth versus reported trajectory" />
      <DivergenceReadout divergence={divergence} alert={alertActive} />
    </Panel>
  );
}

function Legend() {
  return (
    <div className="flex items-center gap-4 text-[10px] tracking-[0.12em] num">
      <span className="flex items-center gap-1.5 text-truth">
        <i className="inline-block h-[2px] w-4 bg-truth" /> TRUTH <span className="text-faint">GOD VIEW</span>
      </span>
      <span className="flex items-center gap-1.5 text-reported">
        <i className="inline-block h-[2px] w-4 bg-reported" /> REPORTED <span className="text-faint">ON THE LINK</span>
      </span>
    </div>
  );
}

interface Divergence {
  total: number;
  unit: string;
  parts: { label: string; value: number; unit: string }[];
}

function measureDivergence(tick: TickMessage | null, schema: SchemaName): Divergence | null {
  if (!tick) return null;
  const d = (k: string) => (tick.reported.channels[k] ?? Number.NaN) - (tick.truth.channels[k] ?? Number.NaN);
  if (schema === "falcon9") {
    const alt = d("altitude") / 1000;
    const dr = d("downrange") / 1000;
    return {
      total: Math.hypot(alt, dr),
      unit: "km",
      parts: [
        { label: "ALT", value: alt, unit: "km" },
        { label: "DR", value: dr, unit: "km" },
        { label: "SPD", value: d("speed"), unit: "m/s" },
      ],
    };
  }
  if (schema === "quad") {
    const n = d("pn");
    const e = d("pe");
    return {
      total: Math.hypot(n, e),
      unit: "m",
      parts: [
        { label: "N", value: n, unit: "m" },
        { label: "E", value: e, unit: "m" },
        { label: "ALT", value: d("alt"), unit: "m" },
      ],
    };
  }
  const v = d("value");
  return { total: Math.abs(v), unit: "norm", parts: [{ label: "VALUE", value: v, unit: "norm" }] };
}

function DivergenceReadout({ divergence, alert }: { divergence: Divergence | null; alert: boolean }) {
  const lying = divergence !== null && divergence.total > 1e-6;
  const color = alert ? "text-alert" : lying ? "text-reported" : "text-muted";
  return (
    <div className="pointer-events-none absolute right-5 top-4 min-w-[180px] border border-line-strong bg-bg/85 px-3 py-2 text-right">
      <div className="label">Reported − truth</div>
      <div className={`num text-[26px] leading-tight ${color}`}>
        {divergence ? formatValue(divergence.total, divergence.total < 10 ? 2 : 1) : "—"}
        <span className="ml-1 text-[12px] text-muted">{divergence?.unit ?? ""}</span>
      </div>
      <div className="num mt-0.5 flex justify-end gap-3 text-[10.5px] text-muted">
        {divergence?.parts.map((p) => (
          <span key={p.label}>
            {p.label} {formatSigned(p.value)}
          </span>
        ))}
      </div>
    </div>
  );
}

function formatSigned(v: number): string {
  if (!Number.isFinite(v)) return "—";
  const s = formatValue(Math.abs(v), Math.abs(v) < 10 ? 2 : 0);
  return `${v < 0 ? "−" : "+"}${s}`;
}
