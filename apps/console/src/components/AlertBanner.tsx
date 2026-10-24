import type { ReactNode } from "react";
import { formatStamp } from "@/lib/format";
import type { AlertBannerView } from "@/lib/stream/reducer";

interface AlertBannerProps {
  view: AlertBannerView;
  framesPassing: boolean;
  running: boolean;
  readout?: ReactNode; // right-hand slot: link validity vs. truth gap, shown in both states
}

/** Always occupies its row so the layout never jumps; turns red only for alerts. */
export function AlertBanner({ view, framesPassing, running, readout }: AlertBannerProps) {
  if (view.active && view.alert) {
    const { alert } = view;
    return (
      <div role="alert" className={`flex h-[58px] items-center gap-5 border-y border-alert pl-5 ${view.live ? "alert-live" : "bg-alert/10"}`}>
        <div className="flex flex-none items-center gap-3">
          <span className="h-3 w-3 rounded-full bg-alert" aria-hidden />
          <div>
            <div className="num text-[17px] font-semibold tracking-[0.16em] text-alert">VALID BUT WRONG</div>
            <div className="num text-[10px] tracking-[0.12em] text-alert/80">
              {formatStamp(alert.t)} · SCORE {alert.score.toFixed(2)} · {alert.detector.toUpperCase()}
              {view.live ? "" : " · CLEARING"}
            </div>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          {alert.spoken ? <div className="truncate text-[14px] font-medium text-text">“{alert.spoken}”</div> : null}
          <div className="truncate text-[12px] text-text/80">{alert.explanation || "Physics inconsistency detected."}</div>
        </div>
        {readout ?? (
          <div className="num flex-none pr-5 text-right text-[10px] leading-tight tracking-[0.1em] text-ok">
            CRC ✓ SEQ ✓
            <br />
            <span className="text-muted">link says fine</span>
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="flex h-[58px] items-center gap-4 bg-panel pl-5">
      <span className={`h-2.5 w-2.5 flex-none rounded-full ${running ? "bg-ok" : "bg-faint"}`} aria-hidden />
      <div className="num flex-none text-[13px] tracking-[0.16em] text-text">{running ? "NOMINAL" : "STANDBY"}</div>
      <div className="min-w-0 flex-1 truncate text-[12px] text-muted">
        {framesPassing
          ? "Every packet on screen passes its checksum. The detector checks whether it is telling the truth."
          : "Link-layer errors present."}
      </div>
      {readout}
    </div>
  );
}
