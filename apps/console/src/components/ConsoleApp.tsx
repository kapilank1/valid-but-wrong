"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { detectorThreshold } from "@/lib/catalog";
import { attackParams, initialControls, resetRequest, startRequest, type RunControls } from "@/lib/controls";
import { resolveSchema } from "@/lib/scenario";
import { missionTime, selectAlertBanner } from "@/lib/stream/reducer";
import { AlertBanner } from "@/components/AlertBanner";
import { AttackLadder } from "@/components/AttackLadder";
import { BenchCard } from "@/components/BenchCard";
import { ChannelStrips } from "@/components/ChannelStrips";
import { DetectorPanel } from "@/components/DetectorPanel";
import { useConsole, useEngine, writeMuted } from "@/components/EngineProvider";
import { EventLog } from "@/components/EventLog";
import { HeroPlot } from "@/components/HeroPlot";
import { LinkPanel } from "@/components/LinkPanel";
import { TopBar } from "@/components/TopBar";

export function ConsoleApp() {
  const engine = useEngine();
  const { stream, connection } = useConsole();
  const { catalog, status, scenario, latest, link } = stream;
  const [controls, setControls] = useState<RunControls | null>(null);
  const [muted, setMuted] = useState(false);

  useEffect(() => setMuted(engine.voice?.isMuted ?? false), [engine]);

  // Seed the controls from the first status, then follow server-side *changes* (e.g. a hot swap
  // from another console) without clobbering a selection the operator hasn't started yet.
  const seen = useRef<{ detector: string; attack: string } | null>(null);
  useEffect(() => {
    if (!status || !catalog) return;
    const prev = seen.current;
    seen.current = { detector: status.detector, attack: status.attack };
    setControls((c) => {
      if (!c) return initialControls(status, catalog);
      const patch: Partial<RunControls> = {};
      if (prev && prev.detector !== status.detector) patch.detector = status.detector;
      if (prev && prev.attack !== status.attack) patch.attack = status.attack;
      return Object.keys(patch).length > 0 ? { ...c, ...patch } : c;
    });
  }, [status, catalog]);

  const schema = resolveSchema(scenario, latest?.truth.schema_id) ?? "falcon9";
  const t = missionTime(stream);
  const running = status?.running ?? false;
  const banner = selectAlertBanner(stream);
  const detector = latest?.verdict.detector ?? status?.detector ?? controls?.detector ?? "base";
  const threshold = detectorThreshold(catalog, detector, schema);

  const patchControls = (patch: Partial<RunControls>) => setControls((c) => (c ? { ...c, ...patch } : c));

  const hotSwapAttack = (name: string) => {
    if (!controls) return;
    patchControls({ attack: name });
    // From the ladder the attack starts now, not at the configured start time.
    void engine.setAttack({ attack: name, attack_params: attackParams({ intensity: controls.intensity, startT: t ?? 0 }) });
  };

  const hotSwapDetector = (name: string) => {
    patchControls({ detector: name });
    void engine.setDetector({ detector: name });
  };

  const toggleMute = () => {
    const next = !muted;
    setMuted(next);
    engine.voice?.setMuted(next);
    writeMuted(next);
  };

  const loadBench = useCallback(() => engine.getBench(), [engine]);

  return (
    <div className="console">
      <TopBar
        connection={connection}
        catalog={catalog}
        scenario={scenario}
        missionT={t}
        running={running}
        controls={controls}
        muted={muted}
        onControls={patchControls}
        onStart={() => controls && void engine.control(startRequest(controls))}
        onStop={() => void engine.control({ action: "stop" })}
        onReset={() => controls && void engine.control(resetRequest(controls))}
        onDetector={hotSwapDetector}
        onToggleMute={toggleMute}
        onRetryLive={() => void engine.start({ forceMock: false })}
      />
      <AlertBanner view={banner} framesPassing={link.frames === link.allOk} running={running} />
      <main className="console-main">
        <div className="col col-left">
          <LinkPanel link={link} l0={latest?.verdict.l0 ?? null} />
          <AttackLadder
            catalog={catalog}
            schema={schema}
            currentAttack={status?.attack ?? controls?.attack ?? "none"}
            attackActive={latest?.attack.active ?? false}
            detector={detector}
            onAttack={hotSwapAttack}
            onDetector={hotSwapDetector}
          />
          <BenchCard load={loadBench} mode={connection.state === "open" ? connection.mode : null} />
        </div>
        <div className="col col-center">
          <HeroPlot buffers={engine.buffers} schema={schema} scenario={scenario} latest={latest} alertActive={banner.live} />
          <ChannelStrips buffers={engine.buffers} schema={schema} latest={latest} threshold={threshold} />
        </div>
        <div className="col col-right">
          <DetectorPanel buffers={engine.buffers} verdict={latest?.verdict ?? null} detector={detector} threshold={threshold} schema={schema} />
          <EventLog log={stream.log} alertCount={stream.alertCount} />
        </div>
      </main>
    </div>
  );
}
