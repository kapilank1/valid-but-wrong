import { DEMO_STEPS, type DemoView } from "@/lib/demo/script";

interface DemoControlProps {
  view: DemoView;
  disabled: boolean;
  onStart(): void;
  onCancel(): void;
}

/** "▶ Demo" runs the pitch; while it runs, a step indicator with a cancel button replaces it. */
export function DemoControl({ view, disabled, onStart, onCancel }: DemoControlProps) {
  const shown = view.active || view.step === DEMO_STEPS;
  const start = (
    <button
      type="button"
      className="btn border-reported/70 text-reported"
      onClick={onStart}
      disabled={disabled}
      title="Runs the pitch: SES-9 at 10×, bias, PPO agent with base silent, hot-swap to hardened"
    >
      {shown ? "↻ Demo" : "▶ Demo"}
    </button>
  );
  if (!shown) return start;
  const done = !view.active;
  return (
    <div className="flex min-w-0 items-center gap-2.5">
      <div
        className={`flex h-[26px] min-w-0 items-center gap-2.5 border px-2.5 ${done ? "border-ok/60" : "border-reported/70"}`}
        aria-live="polite"
      >
        <span className={`num flex-none text-[10px] tracking-[0.16em] ${done ? "text-ok" : "text-reported"}`}>
          DEMO {view.step}/{DEMO_STEPS}
        </span>
        <span className="flex flex-none gap-1" aria-hidden>
          {Array.from({ length: DEMO_STEPS }, (_, i) => (
            <i
              key={i}
              className={`h-1.5 w-3 ${i < view.step ? (done ? "bg-ok" : "bg-reported") : "bg-line-strong"} ${
                i === view.step - 1 && !done ? "animate-pulse" : ""
              }`}
            />
          ))}
        </span>
        <span className="truncate text-[12px] text-text">{view.title}</span>
        {view.detail ? <span className="num flex-none truncate text-[11px] text-muted">{view.detail}</span> : null}
      </div>
      {done ? (
        start
      ) : (
        <button type="button" className="btn h-[26px] px-2" onClick={onCancel}>
          Cancel
        </button>
      )}
    </div>
  );
}
