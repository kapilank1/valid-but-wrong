/**
 * Make server-side HTTPS (the Cartesia TTS proxy) trust the operating system's certificate store as well as Node's
 * bundled roots, the way browsers do.
 *
 * Why: TLS-inspecting antivirus (Norton on the demo laptop) re-signs HTTPS with its own root, which it installs in the
 * OS store but Node doesn't read by default, so every fetch() from the server failed with
 * UNABLE_TO_VERIFY_LEAF_SIGNATURE. Adding the OS store fixes that and changes nothing on a normal machine.
 */
export interface TlsCaApi {
  getCACertificates?: (type: "default" | "system") => string[];
  setDefaultCACertificates?: (certs: string[]) => void;
}

/** Returns how many OS certificates were added (0 when the Node version can't do it or the store is empty). */
export function trustSystemCertificates(tls: TlsCaApi): number {
  if (typeof tls.getCACertificates !== "function" || typeof tls.setDefaultCACertificates !== "function") return 0;
  const system = tls.getCACertificates("system");
  if (system.length === 0) return 0;
  const merged = Array.from(new Set([...tls.getCACertificates("default"), ...system]));
  tls.setDefaultCACertificates(merged);
  return system.length;
}
