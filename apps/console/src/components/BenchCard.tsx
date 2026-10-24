"use client";

import { useEffect, useState } from "react";
import { normalizeBench, type BenchView, type Prf } from "@/lib/bench";
import { formatPercent } from "@/lib/format";
import { Panel } from "@/components/Panel";

export function BenchCard({ load, mode }: { load: () => Promise<unknown>; mode: string | null }) {
  const [view, setView] = useState<BenchView | null | "loading">("loading");

  useEffect(() => {
    if (mode === null) return;
    let cancelled = false;
    load().then((raw) => {
      if (!cancelled) setView(normalizeBench(raw));
    });
    return () => {
      cancelled = true;
    };
  }, [load, mode]);

  const mock = view !== null && view !== "loading" && view.mock;
  return (
    <Panel
      title="Benchmark · NASA SMAP / MSL"
      meta={mock ? <span className="num border border-reported px-1 text-[9.5px] text-reported">MOCK DATA</span> : null}
      bodyClassName="overflow-hidden"
    >
      {view === "loading" ? (
        <div className="num text-[11px] text-faint">loading…</div>
      ) : view === null ? (
        <div className="num text-[11px] text-faint">No bench results yet (GET /api/bench).</div>
      ) : (
        <BenchTable view={view} />
      )}
    </Panel>
  );
}

function BenchTable({ view }: { view: BenchView }) {
  return (
    <div className="flex flex-col gap-2">
      <table className="num w-full text-[11px]">
        <thead>
          <tr className="text-faint">
            <th className="pb-1 text-left font-normal" />
            <th className="pb-1 text-right font-normal">P</th>
            <th className="pb-1 text-right font-normal">R</th>
            <th className="pb-1 text-right font-normal">F1</th>
          </tr>
        </thead>
        <tbody>
          {view.rows.map((row) => (
            <BenchRows key={row.dataset} dataset={row.dataset} ours={row.ours} baseline={row.baseline} />
          ))}
        </tbody>
      </table>
      <div className="flex justify-between text-[10px] text-muted">
        <span className="truncate">vs {view.baselineName ?? "baseline"}</span>
        {view.latencyMs !== null ? <span className="num">latency {view.latencyMs.toFixed(1)} ms</span> : null}
      </div>
    </div>
  );
}

function BenchRows({ dataset, ours, baseline }: { dataset: string; ours: Prf; baseline: Prf | null }) {
  const better = baseline ? ours.f1 - baseline.f1 : null;
  return (
    <>
      <tr className="border-t border-line">
        <td className="pt-1 text-text">
          {dataset} <span className="text-faint">ours</span>
        </td>
        <td className="pt-1 text-right">{formatPercent(ours.precision)}</td>
        <td className="pt-1 text-right">{formatPercent(ours.recall)}</td>
        <td className="pt-1 text-right text-truth">
          {formatPercent(ours.f1)}
          {better !== null ? (
            <span className={`ml-1 text-[9.5px] ${better >= 0 ? "text-ok" : "text-muted"}`}>
              {better >= 0 ? "+" : "−"}
              {Math.abs(better * 100).toFixed(1)}
            </span>
          ) : null}
        </td>
      </tr>
      {baseline ? (
        <tr className="text-muted">
          <td className="pb-1 pl-3 text-[10px]">baseline</td>
          <td className="pb-1 text-right">{formatPercent(baseline.precision)}</td>
          <td className="pb-1 text-right">{formatPercent(baseline.recall)}</td>
          <td className="pb-1 text-right">{formatPercent(baseline.f1)}</td>
        </tr>
      ) : null}
    </>
  );
}
