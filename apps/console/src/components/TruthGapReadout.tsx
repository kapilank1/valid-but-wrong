import type { ReactNode } from "react";
import type { LinkValidity, TruthGap } from "@/lib/truthGap";

interface TruthGapReadoutProps {
  link: LinkValidity;
  gap: TruthGap | null;
  alert: boolean;
}

/** The headline: the link is 100% valid, and the picture is still kilometres wrong. */
export function TruthGapReadout({ link, gap, alert }: TruthGapReadoutProps) {
  const linkColor = link.fraction === null ? "text-muted" : link.fraction === 1 ? "text-ok" : "text-alert";
  const gapColor = !gap || gap.level === "none" ? "text-muted" : alert || gap.level === "large" ? "text-alert" : "text-reported";
  const linkSub =
    link.forged > 0 ? `${link.forgedPassing.toLocaleString("en-US")} forged frames passed` : "sync · crc · seq · time";
  return (
    <div className="flex flex-none items-stretch self-stretch border-l border-line-strong" aria-label="Link validity versus truth">
      <Cell label="Link" sub={linkSub}>
        <span className={linkColor}>{link.text}</span>
        <span className="ml-1.5 text-[12px] tracking-[0.14em] text-muted">VALID</span>
      </Cell>
      <span className="my-2.5 w-px bg-line-strong" aria-hidden />
      <Cell label={`Picture · ${gap?.channel ?? "truth"}`} sub="reported − truth · god view">
        <span className="mr-1.5 text-[12px] tracking-[0.14em] text-muted">OFF BY</span>
        <span className={gapColor}>{gap?.text ?? "—"}</span>
        <span className="ml-1 text-[12px] text-muted">{gap?.unit ?? ""}</span>
      </Cell>
    </div>
  );
}

function Cell({ label, sub, children }: { label: string; sub: string; children: ReactNode }) {
  return (
    <div className="flex w-[196px] flex-col justify-center px-4">
      <div className="label leading-none">{label}</div>
      <div className="num mt-1 flex items-baseline whitespace-nowrap text-[24px] font-semibold leading-none">{children}</div>
      <div className="num mt-1 truncate text-[9.5px] leading-none tracking-[0.08em] text-faint">{sub}</div>
    </div>
  );
}
