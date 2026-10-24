# Module contract

Every module codes against `vbw/schema.py` (types) and the registry functions below. Don't rename these without updating every caller.

## Python registries

```python
# vbw/sources/registry.py
list_scenarios() -> list[dict]          # [{id, label, schema, vehicle_id, rate_hz, duration_s, events, description}]
make_source(scenario_id: str, seed: int | None = None) -> Source

# vbw/attacker/registry.py
list_attacks() -> list[dict]            # [{name, label, description, schemas: [schema names], params: {...defaults}}]
make_attack(name: str, schema: ChannelSchema, **params) -> Attack   # "none" must exist (identity)

# vbw/detector/registry.py
list_detectors() -> list[dict]          # [{name, label, description}]  names: "base", "hardened"
make_detector(name: str, schema: ChannelSchema) -> Detector
```

Scenario ids: `falcon9:<slug>` (slug from data/processed/falcon9/index.json), `quad:<mission>`, `smap:<chan_id>` / `msl:<chan_id>`, `sitl:udp:<port>`.

Attacks are given true frames in order and return the frame the attacker wants the ground to see. They keep `seq`, `t`, `vehicle_id`, `schema_id` unchanged (the codec re-encodes with a valid CRC). Attack params always include `start_t` (mission time the attack begins, default: scenario-dependent) and `intensity` (0..1 scale).

## Pipeline server (FastAPI, port 8000)

- `GET  /api/health` → `{ok: true}`
- `GET  /api/catalog` → `{scenarios: [...], attacks: [...], detectors: [...]}`
- `GET  /api/status` → `Status`
- `POST /api/control` body `{action: "start"|"stop"|"reset", scenario?, attack?, attack_params?, detector?, speed?}` → `Status`
- `POST /api/attack` body `{attack, attack_params?}`: hot-swap the attack mid-run
- `POST /api/detector` body `{detector}`: hot-swap the detector mid-run
- `GET  /api/bench` → contents of `results/bench_summary.json` (or `{available: false}`)
- `WS   /ws` → server pushes JSON messages:

```ts
{ type: "hello", catalog: Catalog, status: Status }
{ type: "scenario", scenario: ScenarioInfo }            // sent on start/reset
{ type: "tick", truth: Frame, reported: Frame, verdict: Verdict, attack: { name: string, active: boolean } }
{ type: "status", status: Status }
{ type: "alert", verdict: Verdict, spoken: string }      // rising edge only, a short line for voice callouts
```

`Status = { running: boolean, scenario: string | null, attack: string, attack_params: object, detector: string, speed: number, t: number, frames: number, alerts: number }`

`Frame` and `Verdict` are `vbw.schema` dataclasses' `to_dict()` output. `Verdict.l0.ok` is included.

The console shows `truth` vs `reported` (a "god view" only the demo has) to make the lie visible.
