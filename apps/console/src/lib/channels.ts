import type { SchemaName } from "@/lib/types";

/** How a channel is labeled and scaled for display (raw SI on the wire, friendlier units on screen). */
export interface ChannelDisplay {
  key: string;
  label: string;
  unit: string;
  scale: number;
}

const RAD = 180 / Math.PI;

export const CHANNEL_DISPLAY: Record<SchemaName, ChannelDisplay[]> = {
  falcon9: [
    { key: "altitude", label: "ALTITUDE", unit: "km", scale: 1e-3 },
    { key: "speed", label: "SPEED", unit: "m/s", scale: 1 },
    { key: "vz", label: "VERT VEL", unit: "m/s", scale: 1 },
    { key: "vx", label: "HORIZ VEL", unit: "m/s", scale: 1 },
    { key: "downrange", label: "DOWNRANGE", unit: "km", scale: 1e-3 },
    { key: "accel", label: "ACCEL", unit: "m/s²", scale: 1 },
    { key: "angle", label: "FLIGHT ANGLE", unit: "deg", scale: 1 },
    { key: "q", label: "DYN PRESSURE", unit: "kPa", scale: 1e-3 },
  ],
  quad: [
    { key: "alt", label: "GPS ALT", unit: "m", scale: 1 },
    { key: "baro_alt", label: "BARO ALT", unit: "m", scale: 1 },
    { key: "pn", label: "NORTH", unit: "m", scale: 1 },
    { key: "pe", label: "EAST", unit: "m", scale: 1 },
    { key: "vn", label: "VEL N", unit: "m/s", scale: 1 },
    { key: "ve", label: "VEL E", unit: "m/s", scale: 1 },
    { key: "groundspeed", label: "GROUNDSPEED", unit: "m/s", scale: 1 },
    { key: "roll", label: "ROLL", unit: "deg", scale: RAD },
    { key: "pitch", label: "PITCH", unit: "deg", scale: RAD },
    { key: "yaw", label: "YAW", unit: "deg", scale: RAD },
    { key: "vu", label: "VEL UP", unit: "m/s", scale: 1 },
    { key: "p", label: "ROLL RATE", unit: "deg/s", scale: RAD },
    { key: "q", label: "PITCH RATE", unit: "deg/s", scale: RAD },
    { key: "r", label: "YAW RATE", unit: "deg/s", scale: RAD },
    { key: "ax", label: "ACCEL X", unit: "m/s²", scale: 1 },
    { key: "ay", label: "ACCEL Y", unit: "m/s²", scale: 1 },
    { key: "az", label: "ACCEL Z", unit: "m/s²", scale: 1 },
    { key: "battery_v", label: "BATTERY", unit: "V", scale: 1 },
  ],
  spacecraft: [{ key: "value", label: "TELEMETRY VALUE", unit: "norm", scale: 1 }],
};

/** Channels shown as strips; the first 8 of each schema's display list. */
export function stripChannels(schema: SchemaName): ChannelDisplay[] {
  return CHANNEL_DISPLAY[schema].slice(0, 8);
}

export function channelDisplay(schema: SchemaName | null, key: string): ChannelDisplay {
  const found = schema ? CHANNEL_DISPLAY[schema].find((c) => c.key === key) : undefined;
  return found ?? { key, label: key.toUpperCase(), unit: "", scale: 1 };
}
