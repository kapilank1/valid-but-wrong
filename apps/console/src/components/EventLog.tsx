import { formatStamp } from "@/lib/format";
import type { LogEntry, LogKind } from "@/lib/stream/reducer";
import { Panel } from "@/components/Panel";

const KIND_STYLE: Record<LogKind, { tag: string; color: string }> = {
  alert: { tag: "ALERT", color: "text-alert" },
  clear: { tag: "CLEAR", color: "text-ok" },
  attack: { tag: "MITM", color: "text-reported" },
  detector: { tag: "DET", color: "text-text" },
  scenario: { tag: "SCN", color: "text-truth" },
  link: { tag: "LINK", color: "text-muted" },
  info: { tag: "INFO", color: "text-muted" },
};

export function EventLog({ log, alertCount }: { log: LogEntry[]; alertCount: number }) {
  return (
    <Panel
      title="Event log"
      meta={<span className={`num text-[10px] ${alertCount > 0 ? "text-alert" : "text-muted"}`}>{alertCount} alerts</span>}
      bodyClassName="overflow-y-auto scroll-thin py-1"
    >
      {log.length === 0 ? (
        <div className="num py-2 text-[11px] text-faint">No events yet.</div>
      ) : (
        <ol className="flex flex-col">
          {log.map((e) => {
            const style = KIND_STYLE[e.kind];
            return (
              <li key={e.id} className="grid grid-cols-[64px_40px_1fr] gap-2 border-b border-line py-1.5 last:border-0">
                <span className="num text-[10.5px] text-muted">{formatStamp(e.t)}</span>
                <span className={`num text-[10px] tracking-[0.08em] ${style.color}`}>{style.tag}</span>
                <div className="min-w-0">
                  <div className={`text-[11.5px] leading-snug ${e.kind === "alert" ? "text-alert" : "text-text"}`}>{e.text}</div>
                  {e.detail ? <div className="text-[10.5px] leading-snug text-muted">{e.detail}</div> : null}
                </div>
              </li>
            );
          })}
        </ol>
      )}
    </Panel>
  );
}
