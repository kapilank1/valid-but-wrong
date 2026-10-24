"use client";

import { createContext, useContext, useEffect, useState, useSyncExternalStore, type ReactNode } from "react";
import { ConsoleEngine, type ConsoleSnapshot } from "@/lib/stream/engine";
import { CalloutQueue } from "@/lib/voice/queue";
import { createSpeaker } from "@/lib/voice/speaker";

const EngineContext = createContext<ConsoleEngine | null>(null);

const MUTE_KEY = "vbw.console.muted";

function readMuted(): boolean {
  try {
    return window.localStorage.getItem(MUTE_KEY) === "1";
  } catch {
    return false;
  }
}

export function writeMuted(muted: boolean): void {
  try {
    window.localStorage.setItem(MUTE_KEY, muted ? "1" : "0");
  } catch {
    // Storage can be blocked (private mode); muting still works for this session.
  }
}

function createEngine(): ConsoleEngine {
  // Constructing touches no browser APIs, so this is safe during server rendering.
  const speaker = typeof window === "undefined" ? null : createSpeaker();
  const voice = speaker ? new CalloutQueue((t) => speaker.speak(t), () => speaker.stop()) : null;
  return new ConsoleEngine(voice);
}

export function EngineProvider({ children }: { children: ReactNode }) {
  const [engine] = useState(createEngine);

  useEffect(() => {
    engine.voice?.setMuted(readMuted());
    const forceMock = new URLSearchParams(window.location.search).get("mock") === "1";
    void engine.start({ forceMock });
    return () => engine.stop();
  }, [engine]);

  return <EngineContext.Provider value={engine}>{children}</EngineContext.Provider>;
}

export function useEngine(): ConsoleEngine {
  const engine = useContext(EngineContext);
  if (!engine) throw new Error("useEngine must be used inside <EngineProvider>");
  return engine;
}

/** Throttled (~5 Hz) view of the stream for text panels. Charts read engine.buffers directly. */
export function useConsole(): ConsoleSnapshot {
  const engine = useEngine();
  return useSyncExternalStore(engine.subscribe, engine.getSnapshot, engine.getServerSnapshot);
}
