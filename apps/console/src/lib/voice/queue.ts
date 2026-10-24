/**
 * Serializes voice callouts so they never talk over each other. Alerts can fire faster than speech,
 * so only the most recent `maxPending` lines wait; older ones are dropped as stale.
 */
export class CalloutQueue {
  private pending: string[] = [];
  private busy = false;
  private muted = false;

  constructor(
    private readonly speak: (text: string) => Promise<void>,
    private readonly stopCurrent: () => void = () => {},
    private readonly maxPending = 2,
  ) {}

  get isMuted(): boolean {
    return this.muted;
  }

  get size(): number {
    return this.pending.length;
  }

  get speaking(): boolean {
    return this.busy;
  }

  enqueue(text: string): void {
    const line = text.trim();
    if (this.muted || line === "" || this.pending.includes(line)) return;
    this.pending.push(line);
    while (this.pending.length > this.maxPending) this.pending.shift();
    void this.drain();
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    if (muted) {
      this.pending = [];
      this.stopCurrent();
    }
  }

  private async drain(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    try {
      while (this.pending.length > 0 && !this.muted) {
        const next = this.pending.shift();
        if (next === undefined) break;
        try {
          await this.speak(next);
        } catch {
          // A failed callout must not wedge the queue.
        }
      }
    } finally {
      this.busy = false;
    }
  }
}
