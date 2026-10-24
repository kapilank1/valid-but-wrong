"use client";

import { drawStrip, SCORE_KEY, STRIP_WINDOW_S } from "@/lib/canvas/strip";
import { useCanvasDraw } from "@/lib/canvas/useCanvasDraw";
import { stripChannels, type ChannelDisplay } from "@/lib/channels";
import { formatValue } from "@/lib/format";
import type { TelemetryBuffers } from "@/lib/stream/buffers";
import type { SchemaName, TickMessage } from "@/lib/types";
import { Panel } from "@/components/Panel";

interface ChannelStripsProps {
  buffers: TelemetryBuffers;
  schema: SchemaName;
  latest: TickMessage | null;
  threshold: number;
}

const SCORE_STRIP: ChannelDisplay = { key: SCORE_KEY, label: "DETECTOR SCORE", unit: "0–1", scale: 1 };

export function ChannelStrips({ buffers, schema, latest, threshold }: ChannelStripsProps) {
  const channels = stripChannels(schema);
  // A single-channel spacecraft gets the detector score beside it instead of an empty grid.
  const strips = channels.length < 4 ? [...channels, SCORE_STRIP] : channels;
  const cols = strips.length < 4 ? strips.length : 4;

  return (
    <Panel
      title="Channel strips · reported vs truth"
      meta={<span className="label text-faint">rolling {STRIP_WINDOW_S}s</span>}
      bodyClassName="p-0"
    >
      <div
        className="grid h-full gap-px bg-line"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gridAutoRows: "minmax(0, 1fr)" }}
      >
        {strips.map((c) => (
          <Strip key={`${schema}:${c.key}`} buffers={buffers} display={c} latest={latest} threshold={threshold} schema={schema} />
        ))}
      </div>
    </Panel>
  );
}

interface StripProps {
  buffers: TelemetryBuffers;
  display: ChannelDisplay;
  latest: TickMessage | null;
  threshold: number;
  schema: SchemaName;
}

function Strip({ buffers, display, latest, threshold, schema }: StripProps) {
  const isScore = display.key === SCORE_KEY;
  const canvasRef = useCanvasDraw(
    buffers,
    (surface, theme) =>
      drawStrip(surface, theme, buffers, { channel: display.key, scale: display.scale, threshold: isScore ? threshold : undefined }),
    `${schema}:${display.key}:${threshold}`,
  );
  const reported = isScore ? latest?.verdict.score : scaled(latest?.reported.channels[display.key], display.scale);
  const truth = isScore ? undefined : scaled(latest?.truth.channels[display.key], display.scale);
  const delta = reported !== undefined && truth !== undefined ? reported - truth : undefined;
  const lying = delta !== undefined && Math.abs(delta) > Math.max(1e-9, Math.abs(truth ?? 0) * 1e-6);

  return (
    <div className="flex min-h-[110px] min-w-0 flex-col bg-panel px-2 pb-1 pt-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span className="label truncate">
          {display.label} <span className="text-faint normal-case tracking-normal">{display.unit}</span>
        </span>
        {lying ? (
          <span className="num text-[10px] text-reported">
            Δ{delta > 0 ? "+" : "−"}
            {formatValue(Math.abs(delta))}
          </span>
        ) : null}
      </div>
      <div className="num flex items-baseline gap-2 text-[13px] leading-5">
        <span className={isScore ? "text-text" : "text-reported"}>{formatValue(reported)}</span>
        {!isScore ? <span className="text-[11px] text-truth">{formatValue(truth)}</span> : null}
      </div>
      <canvas ref={canvasRef} className="mt-0.5 min-h-0 w-full flex-1" />
    </div>
  );
}

function scaled(v: number | undefined, scale: number): number | undefined {
  return typeof v === "number" ? v * scale : undefined;
}
