import type { Catalog, SchemaName } from "@/lib/types";

export interface Availability {
  available: boolean;
  reason?: string;
}

/** Whether an attack can be launched against the current vehicle, per the server's catalog. */
export function attackAvailability(catalog: Catalog | null, name: string, schema: SchemaName | null): Availability {
  if (!catalog) return { available: false, reason: "no catalog yet" };
  const attack = catalog.attacks.find((a) => a.name === name);
  if (!attack) return { available: false, reason: "not in server catalog" };
  if (attack.available === false) return { available: false, reason: "unavailable on server" };
  if (schema && attack.schemas.length > 0 && !attack.schemas.includes(schema)) {
    return { available: false, reason: `not supported for ${schema}` };
  }
  return { available: true };
}

export function detectorAvailability(catalog: Catalog | null, name: string): Availability {
  if (!catalog) return { available: false, reason: "no catalog yet" };
  const detector = catalog.detectors.find((d) => d.name === name);
  if (!detector) return { available: false, reason: "not in server catalog" };
  if (detector.available === false) return { available: false, reason: "unavailable on server" };
  return { available: true };
}

export const DEFAULT_THRESHOLD = 0.5;

/** Alert threshold for the gauge. Not in the contract yet, so read it if the catalog provides one. */
export function detectorThreshold(catalog: Catalog | null, name: string): number {
  const threshold = catalog?.detectors.find((x) => x.name === name)?.threshold;
  return typeof threshold === "number" ? threshold : DEFAULT_THRESHOLD;
}
