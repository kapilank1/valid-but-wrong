"use client";

import { useEffect, useRef } from "react";
import { onFrame } from "@/lib/canvas/frameLoop";
import { prepare, readTheme, type Surface, type Theme } from "@/lib/canvas/draw";
import type { TelemetryBuffers } from "@/lib/stream/buffers";

export type DrawFn = (surface: Surface, theme: Theme) => void;

/**
 * Redraws a canvas on the shared animation frame, but only when the buffers changed, the canvas was
 * resized, or `key` (anything else the drawing depends on) changed.
 */
export function useCanvasDraw(buffers: TelemetryBuffers, draw: DrawFn, key: string) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawRef = useRef(draw);
  const keyRef = useRef(key);
  useEffect(() => {
    drawRef.current = draw;
    keyRef.current = key;
  });

  useEffect(() => {
    let lastVersion = -1;
    let lastKey = "";
    return onFrame(() => {
      const canvas = canvasRef.current;
      if (!canvas || canvas.clientWidth === 0) return;
      const surface = prepare(canvas);
      if (!surface) return;
      if (!surface.resized && buffers.version === lastVersion && keyRef.current === lastKey) return;
      lastVersion = buffers.version;
      lastKey = keyRef.current;
      const { ctx, width, height } = surface;
      ctx.clearRect(0, 0, width, height);
      drawRef.current(surface, readTheme());
    });
  }, [buffers]);

  return canvasRef;
}
