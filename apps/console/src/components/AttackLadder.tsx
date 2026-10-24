import { attackAvailability, detectorAvailability } from "@/lib/catalog";
import type { Catalog, SchemaName } from "@/lib/types";
import { Panel } from "@/components/Panel";

const RUNGS: { name: string; label: string; note: string }[] = [
  { name: "bias", label: "Bias", note: "constant offset · L1 bait" },
  { name: "drift", label: "Drift", note: "slow ramp · L2 territory" },
  { name: "replay", label: "Replay", note: "old frames, fresh seq" },
  { name: "coordinated", label: "Coordinated", note: "consistent kinematics" },
  { name: "ppo", label: "PPO agent", note: "learned evasion" },
];

interface AttackLadderProps {
  catalog: Catalog | null;
  schema: SchemaName | null;
  currentAttack: string;
  attackActive: boolean;
  detector: string;
  onAttack(name: string): void;
  onDetector(name: string): void;
}

export function AttackLadder({ catalog, schema, currentAttack, attackActive, detector, onAttack, onDetector }: AttackLadderProps) {
  const hardened = detector === "hardened";
  const hardenable = detectorAvailability(catalog, "hardened");
  return (
    <Panel
      title="Attack ladder · red team"
      meta={
        <button type="button" className="btn h-[18px] px-1.5 text-[9.5px]" onClick={() => onAttack("none")} disabled={currentAttack === "none"}>
          Clear
        </button>
      }
    >
      <ol className="flex flex-col gap-1">
        {RUNGS.map((r, i) => {
          const avail = attackAvailability(catalog, r.name, schema);
          const selected = currentAttack === r.name;
          return (
            <li key={r.name}>
              <button
                type="button"
                className="btn w-full justify-start px-2"
                aria-pressed={selected}
                disabled={!avail.available}
                title={avail.available ? `POST /api/attack {attack: "${r.name}"}` : avail.reason}
                onClick={() => onAttack(r.name)}
              >
                <span className="num w-4 text-faint">{i + 1}</span>
                <span className={`w-[92px] text-left ${selected ? "text-reported" : ""}`}>{r.label}</span>
                <span className="flex-1 truncate text-left normal-case tracking-normal text-[10.5px] text-muted">
                  {avail.available ? r.note : avail.reason}
                </span>
                {selected ? (
                  <span className={`h-1.5 w-1.5 rounded-full ${attackActive ? "bg-reported" : "bg-faint"}`} aria-label={attackActive ? "active" : "armed"} />
                ) : null}
              </button>
            </li>
          );
        })}
      </ol>
      <button
        type="button"
        className="btn mt-2 w-full"
        aria-pressed={hardened}
        disabled={!hardenable.available}
        title={hardenable.available ? "POST /api/detector — hot-swap without stopping the run" : hardenable.reason}
        onClick={() => onDetector(hardened ? "base" : "hardened")}
      >
        {hardened ? "Hardened detector in · revert to base" : "Harden detector"}
      </button>
    </Panel>
  );
}
