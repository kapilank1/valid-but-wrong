import { describe, expect, it } from "vitest";
import { trustSystemCertificates, type TlsCaApi } from "@/lib/systemCa";

function fakeTls(defaults: string[], system: string[]) {
  let installed: string[] | null = null;
  const tls: TlsCaApi = {
    getCACertificates: (type) => (type === "system" ? system : defaults),
    setDefaultCACertificates: (certs) => {
      installed = certs;
    },
  };
  return { tls, installed: () => installed };
}

describe("trustSystemCertificates", () => {
  it("adds the OS store to Node's bundled roots without duplicates", () => {
    const { tls, installed } = fakeTls(["A", "B"], ["B", "NORTON"]);
    expect(trustSystemCertificates(tls)).toBe(2);
    expect(installed()).toEqual(["A", "B", "NORTON"]);
  });

  it("does nothing on Node versions without the CA API or with an empty OS store", () => {
    expect(trustSystemCertificates({})).toBe(0);
    const { tls, installed } = fakeTls(["A"], []);
    expect(trustSystemCertificates(tls)).toBe(0);
    expect(installed()).toBeNull();
  });
});
