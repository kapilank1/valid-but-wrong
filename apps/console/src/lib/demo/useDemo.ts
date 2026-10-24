"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  IDLE_DEMO,
  cancelDemo,
  demoView,
  isDemoActive,
  observeDemo,
  startDemo,
  stepDemo,
  type DemoCommand,
  type DemoState,
  type DemoStep,
  type DemoView,
} from "@/lib/demo/script";
import type { ConsoleEngine } from "@/lib/stream/engine";
import type { StreamState } from "@/lib/stream/reducer";

export interface DemoControls {
  state: DemoState;
  view: DemoView;
  start(): void;
  cancel(): void;
}

/**
 * Thin React wrapper around the pure demo script: on every stream snapshot (~5 Hz) it steps the script and sends
 * the commands it returns, in order. It never steps while commands are still in flight.
 */
export function useDemo(engine: ConsoleEngine, stream: StreamState): DemoControls {
  const [state, setState] = useState<DemoState>(IDLE_DEMO);
  const stateRef = useRef<DemoState>(IDLE_DEMO);
  const busy = useRef(false);

  const send = useCallback(
    async (commands: DemoCommand[]) => {
      for (const c of commands) {
        if (c.kind === "control") await engine.control(c.body);
        else if (c.kind === "attack") await engine.setAttack(c.body);
        else await engine.setDetector(c.body);
      }
    },
    [engine],
  );

  const apply = useCallback(
    (step: DemoStep) => {
      stateRef.current = step.state;
      setState(step.state);
      if (step.commands.length === 0) return;
      busy.current = true;
      void send(step.commands).finally(() => {
        busy.current = false;
      });
    },
    [send],
  );

  useEffect(() => {
    if (busy.current || !isDemoActive(stateRef.current)) return;
    const step = stepDemo(stateRef.current, observeDemo(stream));
    if (step.state !== stateRef.current || step.commands.length > 0) apply(step);
  }, [stream, apply]);

  const start = useCallback(() => apply(startDemo()), [apply]);
  const cancel = useCallback(() => apply(cancelDemo(stateRef.current)), [apply]);

  return { state, view: demoView(state, observeDemo(stream)), start, cancel };
}
