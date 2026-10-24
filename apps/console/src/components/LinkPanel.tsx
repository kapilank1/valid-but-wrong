import { formatCount, formatPercent } from "@/lib/format";
import { crcPassRate, type LinkCounters } from "@/lib/stream/reducer";
import type { L0Result } from "@/lib/types";
import { Panel } from "@/components/Panel";

interface LinkPanelProps {
  link: LinkCounters;
  l0: L0Result | null;
}

const CHECKS: { key: keyof Pick<LinkCounters, "sync" | "crc" | "seq" | "time">; flag: keyof L0Result; label: string; detail: string }[] = [
  { key: "sync", flag: "sync_ok", label: "SYNC", detail: "ASM 1ACFFC1D" },
  { key: "crc", flag: "crc_ok", label: "CRC", detail: "CRC-16/CCITT" },
  { key: "seq", flag: "seq_ok", label: "SEQ", detail: "monotonic u16" },
  { key: "time", flag: "time_ok", label: "TIME", detail: "mission clock" },
];

/** L0 link integrity. During an attack these stay green, and that is the point. */
export function LinkPanel({ link, l0 }: LinkPanelProps) {
  const rate = crcPassRate(link);
  const allGood = link.frames === link.allOk;

  return (
    <Panel title="L0 · Link integrity" meta={<span className="num text-[10px] text-muted">{formatCount(link.frames)} frames</span>}>
      <ul className="grid grid-cols-2 gap-px bg-line">
        {CHECKS.map((c) => {
          const ok = l0 ? Boolean(l0[c.flag]) : true;
          const passed = link[c.key];
          return (
            <li key={c.key} className="flex items-center justify-between bg-panel px-2 py-1.5">
              <div>
                <div className="num text-[12px] tracking-[0.12em] text-text">{c.label}</div>
                <div className="text-[9.5px] text-faint num">{c.detail}</div>
              </div>
              <div className="text-right">
                <div className={`num text-[15px] leading-none ${ok ? "text-ok" : "text-alert"}`} aria-label={ok ? "pass" : "fail"}>
                  {ok ? "✓" : "✗"}
                </div>
                <div className="num text-[10px] text-muted">{formatCount(passed)}</div>
              </div>
            </li>
          );
        })}
      </ul>

      <div className="mt-3 flex items-end justify-between">
        <div>
          <div className="label">Frames passing CRC</div>
          <div className={`num text-[30px] leading-none ${rate === 1 ? "text-ok" : "text-alert"}`}>{formatPercent(rate, rate === 1 ? 0 : 1)}</div>
        </div>
        <div className="text-right">
          <div className="label">Forged on link</div>
          <div className={`num text-[20px] leading-none ${link.forged > 0 ? "text-reported" : "text-muted"}`}>{formatCount(link.forged)}</div>
          <div className="num text-[9.5px] text-faint">god view</div>
        </div>
      </div>
      <p className="mt-2 text-[10.5px] leading-snug text-muted">
        {link.forged > 0 && allGood
          ? `All ${formatCount(link.forgedPassing)} forged frames passed every link check.`
          : "Integrity checks prove the bits arrived intact, not that they are true."}
      </p>
    </Panel>
  );
}
