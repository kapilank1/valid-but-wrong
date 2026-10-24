import { applyMockAttack } from "@/lib/mock/attacks";
import { mockCatalog } from "@/lib/mock/catalog";
import { MockDetector, spokenLine } from "@/lib/mock/detector";
import { Rng } from "@/lib/mock/rng";
import { mockTruthModels, type TruthModel } from "@/lib/mock/vehicles";
import type {
  AttackParams,
  AttackRequest,
  Catalog,
  ControlRequest,
  DetectorRequest,
  Frame,
  ServerMessage,
  Status,
} from "@/lib/types";

/** Per-schema measurement noise (1σ) added to the true signal, so "truth" looks like a real sensor. */
const NOISE: Record<string, Record<string, number>> = {
  quad: { pn: 0.05, pe: 0.05, alt: 0.05, baro_alt: 0.15, vn: 0.02, ve: 0.02 },
  spacecraft: { value: 0.01 },
};

/** Never emit more than this many frames per advance() so a stalled tab can't spiral. */
const MAX_FRAMES_PER_ADVANCE = 60;

/**
 * Deterministic in-browser imitation of the pipeline server. It speaks the same contract
 * (control/attack/detector requests in, ServerMessages out), so the console can't tell the difference.
 * Wall-clock driving lives in MockTransport; this class only advances when told to.
 */
export class MockSim {
  readonly catalog: Catalog;
  private readonly models: TruthModel[];
  private readonly rng: Rng;
  private model: TruthModel;
  private detector: MockDetector;
  private status: Status;
  private seq = 0;
  private pending = 0; // mission seconds accumulated but not yet emitted as frames
  private lastAlert = false;
  private statusClock = 0;

  constructor(seed = 7) {
    this.rng = new Rng(seed);
    this.models = mockTruthModels();
    this.catalog = mockCatalog(this.models);
    const first = this.models[0];
    if (!first) throw new Error("mock has no scenarios");
    this.model = first;
    this.detector = new MockDetector("base", first, this.rng);
    this.status = {
      running: false,
      scenario: first.info.id,
      attack: "none",
      attack_params: this.defaultParams(first),
      detector: "base",
      speed: 1,
      t: 0,
      frames: 0,
      alerts: 0,
    };
  }

  hello(): ServerMessage[] {
    return [
      { type: "hello", catalog: this.catalog, status: this.snapshot() },
      { type: "scenario", scenario: this.model.info },
    ];
  }

  getStatus(): Status {
    return this.snapshot();
  }

  control(req: ControlRequest): ServerMessage[] {
    const out: ServerMessage[] = [];
    const wantScenario = req.scenario ?? this.status.scenario ?? this.model.info.id;
    if (req.action === "reset" || wantScenario !== this.model.info.id) {
      out.push(...this.load(wantScenario));
    }
    if (req.attack !== undefined) this.status.attack = req.attack;
    if (req.attack_params !== undefined) this.status.attack_params = { ...this.status.attack_params, ...req.attack_params };
    if (req.detector !== undefined) this.swapDetector(req.detector);
    if (req.speed !== undefined && req.speed > 0) this.status.speed = req.speed;
    if (req.action === "start") {
      // Restart from zero if the previous run played to the end.
      if (this.model.info.duration_s !== null && this.status.t >= this.model.info.duration_s) out.push(...this.load(wantScenario));
      this.status.running = true;
    }
    if (req.action === "stop" || req.action === "reset") this.status.running = false;
    out.push(this.statusMessage());
    return out;
  }

  setAttack(req: AttackRequest): ServerMessage[] {
    this.status.attack = req.attack;
    if (req.attack_params) this.status.attack_params = { ...this.status.attack_params, ...req.attack_params };
    return [this.statusMessage()];
  }

  setDetector(req: DetectorRequest): ServerMessage[] {
    this.swapDetector(req.detector);
    return [this.statusMessage()];
  }

  /** Advance by `wallSeconds` of real time (scaled by playback speed) and return the emitted messages. */
  advance(wallSeconds: number): ServerMessage[] {
    if (!this.status.running) return [];
    const out: ServerMessage[] = [];
    const period = 1 / this.model.info.rate_hz;
    this.pending += wallSeconds * this.status.speed;
    let emitted = 0;
    while (this.pending >= period && emitted < MAX_FRAMES_PER_ADVANCE) {
      this.pending -= period;
      emitted += 1;
      out.push(...this.step(this.status.t + period));
      const end = this.model.info.duration_s;
      if (end !== null && this.status.t >= end) {
        this.status.running = false;
        this.pending = 0;
        out.push(this.statusMessage());
        return out;
      }
    }
    if (emitted === MAX_FRAMES_PER_ADVANCE) this.pending = 0;
    this.statusClock += wallSeconds;
    if (this.statusClock >= 1) {
      this.statusClock = 0;
      out.push(this.statusMessage());
    }
    return out;
  }

  private step(t: number): ServerMessage[] {
    this.status.t = t;
    this.status.frames += 1;
    this.seq = (this.seq + 1) & 0xffff;
    const truthChannels = this.addNoise(this.model.sample(t));
    const truth: Frame = {
      vehicle_id: this.model.info.vehicle_id,
      schema_id: this.model.schema.schema_id,
      seq: this.seq,
      t,
      channels: truthChannels,
    };
    const params = this.status.attack_params;
    const attack = applyMockAttack(
      this.status.attack,
      { model: this.model, t, startT: num(params.start_t, this.model.defaultAttackStart), intensity: num(params.intensity, 0.5) },
      truthChannels,
    );
    // seq, t, vehicle_id and schema_id are untouched; the codec would recompute a valid CRC.
    const reported: Frame = { ...truth, channels: attack.channels };
    const verdict = this.detector.update(truth, reported, this.status.attack);
    const out: ServerMessage[] = [
      { type: "tick", truth, reported, verdict, attack: { name: this.status.attack, active: attack.active } },
    ];
    if (verdict.alert && !this.lastAlert) {
      this.status.alerts += 1;
      out.push({ type: "alert", verdict, spoken: spokenLine(verdict) });
    }
    this.lastAlert = verdict.alert;
    return out;
  }

  private load(scenarioId: string): ServerMessage[] {
    const model = this.models.find((m) => m.info.id === scenarioId) ?? this.model;
    const changed = model !== this.model;
    this.model = model;
    this.detector = new MockDetector(this.status.detector, model, this.rng);
    this.seq = 0;
    this.pending = 0;
    this.lastAlert = false;
    this.status.scenario = model.info.id;
    this.status.t = 0;
    this.status.frames = 0;
    this.status.alerts = 0;
    if (changed) this.status.attack_params = { ...this.status.attack_params, start_t: model.defaultAttackStart };
    return [{ type: "scenario", scenario: model.info }];
  }

  private swapDetector(name: string): void {
    if (name === this.status.detector) return;
    this.status.detector = name;
    // Hot swap keeps L1 history; only the model behind L2/L3 changes.
    this.detector.name = name;
  }

  private addNoise(channels: Record<string, number>): Record<string, number> {
    const sigmas = NOISE[this.model.schema.name];
    if (!sigmas) return channels;
    const out = { ...channels };
    for (const [k, s] of Object.entries(sigmas)) {
      const v = out[k];
      if (typeof v === "number") out[k] = v + s * this.rng.gauss();
    }
    return out;
  }

  private defaultParams(model: TruthModel): AttackParams {
    return { start_t: model.defaultAttackStart, intensity: 0.5 };
  }

  private snapshot(): Status {
    return { ...this.status, attack_params: { ...this.status.attack_params } };
  }

  private statusMessage(): ServerMessage {
    return { type: "status", status: this.snapshot() };
  }
}

function num(v: unknown, fallback: number): number {
  const n = typeof v === "string" ? Number(v) : v;
  return typeof n === "number" && Number.isFinite(n) ? n : fallback;
}
