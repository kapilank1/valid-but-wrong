import type { ReactNode } from "react";
import { detectorAvailability } from "@/lib/catalog";
import { SPEEDS, type RunControls } from "@/lib/controls";
import { formatMissionClock } from "@/lib/format";
import type { ConnectionInfo } from "@/lib/transport/types";
import type { Catalog, ScenarioInfo } from "@/lib/types";

interface TopBarProps {
  connection: ConnectionInfo;
  catalog: Catalog | null;
  scenario: ScenarioInfo | null;
  missionT: number | null;
  running: boolean;
  controls: RunControls | null;
  muted: boolean;
  onControls(patch: Partial<RunControls>): void;
  onStart(): void;
  onStop(): void;
  onReset(): void;
  onDetector(name: string): void;
  onToggleMute(): void;
  onRetryLive(): void;
  demo?: ReactNode; // the one-click pitch control, centered in the title row
}

export function TopBar(props: TopBarProps) {
  const { connection, scenario, missionT, muted, onToggleMute, onRetryLive, demo } = props;
  return (
    <header className="bg-panel">
      <div className="flex h-[42px] items-center gap-5 border-b border-line px-4">
        <div className="flex items-center gap-2.5">
          <Mark />
          <div className="leading-none">
            <div className="num text-[13px] font-semibold tracking-[0.22em] text-text">VALID-BUT-WRONG</div>
            <div className="num mt-0.5 text-[9px] tracking-[0.14em] text-faint">LINK INTEGRITY ≠ TRUTH</div>
          </div>
        </div>
        <div className="h-6 w-px bg-line-strong" />
        <div className="min-w-0 max-w-[280px] flex-none leading-none">
          <div className="truncate text-[13px] text-text">{scenario?.label ?? "No scenario loaded"}</div>
          <div className="num mt-0.5 truncate text-[10px] text-muted">
            {scenario ? `${scenario.vehicle_id} · ${scenario.rate_hz} Hz · ${scenario.id}` : "—"}
          </div>
        </div>
        <div className="flex min-w-0 flex-1 justify-center">{demo}</div>
        <div className="flex flex-none items-center gap-3">
          <ConnectionPill connection={connection} onRetryLive={onRetryLive} />
          <button type="button" className="btn w-[74px]" aria-pressed={!muted} onClick={onToggleMute} title="Voice callouts">
            {muted ? "Voice off" : "Voice on"}
          </button>
          <div className="num min-w-[150px] text-right text-[22px] leading-none tracking-[0.04em] text-text" aria-label="Mission clock">
            {formatMissionClock(missionT)}
          </div>
        </div>
      </div>
      <ControlStrip {...props} />
    </header>
  );
}

function Mark() {
  // Two offset tracks: the truth and the lie.
  return (
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden>
      <path d="M2 19 C 7 15, 10 8, 20 4" fill="none" stroke="var(--color-truth)" strokeWidth="1.8" />
      <path d="M2 19 C 8 16, 13 12, 20 10" fill="none" stroke="var(--color-reported)" strokeWidth="1.8" strokeDasharray="2.5 2" />
    </svg>
  );
}

function ConnectionPill({ connection, onRetryLive }: { connection: ConnectionInfo; onRetryLive(): void }) {
  if (connection.mode === "mock") {
    const fallback = !connection.reason?.includes("?mock=1");
    return (
      <div className="flex items-center gap-2" title={connection.reason}>
        <span className="num border border-reported bg-reported/10 px-2 py-0.5 text-[11px] font-semibold tracking-[0.2em] text-reported">MOCK</span>
        <span className="num max-w-[220px] truncate text-[10px] text-muted">{connection.reason}</span>
        {fallback ? (
          <button type="button" className="btn h-[20px] px-1.5 text-[9.5px]" onClick={onRetryLive}>
            Retry live
          </button>
        ) : null}
      </div>
    );
  }
  const label =
    connection.state === "open"
      ? "LIVE"
      : connection.state === "reconnecting"
        ? `RECONNECTING${connection.retryInMs ? ` ${(connection.retryInMs / 1000).toFixed(1)}s` : ""}`
        : connection.state === "connecting"
          ? "CONNECTING"
          : "OFFLINE";
  const dot = connection.state === "open" ? "bg-ok" : "bg-muted animate-pulse";
  return (
    <span className="num flex items-center gap-2 border border-line-strong px-2 py-0.5 text-[11px] tracking-[0.18em] text-text" title={connection.reason}>
      <i className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  );
}

// The catalog labels ("Physics + learned (L1+L2)") are too long for the strip at 1440 px; the tooltip keeps them.
const SHORT_DETECTOR_LABELS: Record<string, string> = { base: "Base · L1+L2", hardened: "Hardened · +L3" };

function ControlStrip({ catalog, controls, running, onControls, onStart, onStop, onReset, onDetector }: TopBarProps) {
  const disabled = !controls || !catalog;
  const c = controls;
  return (
    <div className="flex h-[40px] items-center gap-2 overflow-x-auto px-4 scroll-thin">
      <Field label="Scenario">
        <select
          className="ctl w-[176px]"
          value={c?.scenario ?? ""}
          disabled={disabled}
          onChange={(e) => onControls({ scenario: e.target.value })}
          aria-label="Scenario"
        >
          {catalog?.scenarios.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label}
            </option>
          ))}
        </select>
      </Field>
      <Divider />
      <Field label="Attack">
        <select
          className="ctl w-[116px]"
          value={c?.attack ?? "none"}
          disabled={disabled}
          onChange={(e) => onControls({ attack: e.target.value })}
          aria-label="Attack"
        >
          {catalog?.attacks.map((a) => (
            <option key={a.name} value={a.name} disabled={a.available === false}>
              {a.label}
            </option>
          ))}
        </select>
      </Field>
      <Field label="Intensity">
        <input
          type="range"
          className="ctl-range w-[68px]"
          min={0}
          max={1}
          step={0.05}
          value={c?.intensity ?? 0.5}
          disabled={disabled}
          onChange={(e) => onControls({ intensity: Number(e.target.value) })}
          aria-label="Attack intensity"
        />
        <span className="num w-8 text-[11px] text-text">{(c?.intensity ?? 0.5).toFixed(2)}</span>
      </Field>
      <Field label="Start T+">
        <input
          type="number"
          className="ctl w-[62px] px-1.5"
          min={0}
          step={5}
          value={c?.startT ?? 60}
          disabled={disabled}
          onChange={(e) => onControls({ startT: Number(e.target.value) })}
          aria-label="Attack start time in seconds"
        />
        <span className="num text-[10px] text-faint">s</span>
      </Field>
      <Divider />
      <Field label="Detector">
        <div className="seg" role="group" aria-label="Detector">
          {(catalog?.detectors ?? []).map((d) => {
            const avail = detectorAvailability(catalog, d.name);
            return (
              <button
                key={d.name}
                type="button"
                className="btn"
                aria-pressed={c?.detector === d.name}
                disabled={disabled || !avail.available}
                title={avail.available ? `${d.label}: ${d.description}` : avail.reason}
                onClick={() => onDetector(d.name)}
              >
                {SHORT_DETECTOR_LABELS[d.name] ?? d.label}
              </button>
            );
          })}
        </div>
      </Field>
      <Divider />
      <div className="flex items-center gap-1.5">
        <button type="button" className="btn border-text" onClick={onStart} disabled={disabled || running}>
          ▶ Start
        </button>
        <button type="button" className="btn" onClick={onStop} disabled={disabled || !running}>
          ■ Stop
        </button>
        <button type="button" className="btn" onClick={onReset} disabled={disabled}>
          ↺ Reset
        </button>
      </div>
      <Field label="Speed">
        <select
          className="ctl w-[58px]"
          value={c?.speed ?? 1}
          disabled={disabled}
          onChange={(e) => onControls({ speed: Number(e.target.value) })}
          aria-label="Playback speed (applies on start)"
          title="Applies on Start / Reset"
        >
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
      </Field>
    </div>
  );
}

// A div, not a <label>: clicking a label would "click" the first button inside the detector toggle.
function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-none items-center gap-2">
      <span className="label">{label}</span>
      {children}
    </div>
  );
}

function Divider() {
  return <span className="h-5 w-px flex-none bg-line-strong" aria-hidden />;
}
