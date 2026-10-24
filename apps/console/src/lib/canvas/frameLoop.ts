/**
 * One shared requestAnimationFrame loop for every canvas on the page. Each subscriber decides on
 * its own whether anything changed (buffer version, size) before drawing.
 */
type FrameCallback = (now: number) => void;

const callbacks = new Set<FrameCallback>();
let handle: number | null = null;

function tick(now: number): void {
  for (const cb of callbacks) {
    try {
      cb(now);
    } catch (err) {
      console.error("frame callback failed", err);
    }
  }
  handle = callbacks.size > 0 ? requestAnimationFrame(tick) : null;
}

export function onFrame(cb: FrameCallback): () => void {
  callbacks.add(cb);
  if (handle === null && typeof requestAnimationFrame !== "undefined") handle = requestAnimationFrame(tick);
  return () => {
    callbacks.delete(cb);
    if (callbacks.size === 0 && handle !== null) {
      cancelAnimationFrame(handle);
      handle = null;
    }
  };
}
